#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Blend your NeMo Gym task rows into the Nemotron 3 Nano RL blend without breaking its curriculum.

Why this exists: nvidia/Nemotron-3-Nano-RL-Training-Blend is ordered from easy (high pass rate)
to hard, and the Nano3 GRPO config reads it in order (`data.shuffle: False`). Appending your rows
to the end would put all of them in the last, hardest steps. Shuffling would destroy the
curriculum for everyone. This script does what the tech report describes instead: a fixed
share of each batch comes from your data, and within your data difficulty also goes from easy
to hard.

    mine  (sorted by pass_rate, high -> low) ──┐
                                               ├─ proportional merge ─> train.jsonl
    nano3 (already ordered easy -> hard)   ────┘                        val.jsonl
                                                                        blend_report.json

Inputs:
  --nano3   resolved Nano3 train JSONL (output of `nemotron nano3 data prep rl`, or of the
            dataset's create_nanov3_jsonl.py). Streamed, so 7 GB is fine.
  --mine    your task rows (from convert_trajectories.py rl). Each row needs `pass_rate`,
            measured by profiling your SFT checkpoint (README step 4). With --allow-historical,
            `historical_pass_rate` is used when `pass_rate` is missing.

Examples:
  # Blend: 15% of every batch is yours, keep all Nano3 rows
  python blend_rl.py --nano3 train-full.jsonl --mine my_rl_tasks.jsonl --ratio 0.15 --out-dir blend/

  # "On top of" a finished RL run: mostly yours, plus a replay slice of Nano3 to prevent regressions
  python blend_rl.py --nano3 train-full.jsonl --mine my_rl_tasks.jsonl --ratio 0.6 \
      --nano3-rows 20000 --out-dir stage2b/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Iterator


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def evenly_spaced(n_total: int, n_pick: int) -> list[int]:
    """n_pick indices spread evenly over range(n_total), so every difficulty band is represented."""
    if n_pick <= 0 or n_total <= 0:
        return []
    n_pick = min(n_pick, n_total)
    return sorted({int((k + 0.5) * n_total / n_pick) for k in range(n_pick)})


def keep_mask(n_total: int, n_keep: int) -> Iterator[bool]:
    """Stream of booleans that keeps n_keep of n_total items, evenly, preserving order."""
    for i in range(n_total):
        yield math.floor((i + 1) * n_keep / n_total) > math.floor(i * n_keep / n_total)


def pass_rate_of(row: dict[str, Any], allow_historical: bool) -> float:
    if row.get("pass_rate") is not None:
        return float(row["pass_rate"])
    if allow_historical and row.get("historical_pass_rate") is not None:
        return float(row["historical_pass_rate"])
    raise SystemExit(
        f"row {row.get('task_id', '?')} has no pass_rate. Profile your SFT checkpoint first "
        "(README step 4) or pass --allow-historical to use historical_pass_rate."
    )


def validate_row(row: dict[str, Any], where: str) -> None:
    if "responses_create_params" not in row and "_hf_placeholder" not in row:
        raise SystemExit(f"{where}: row has no responses_create_params")
    ref = row.get("agent_ref") or {}
    if ref.get("type") != "responses_api_agents" or not ref.get("name"):
        raise SystemExit(f"{where}: row needs agent_ref {{type: responses_api_agents, name: <agent>}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nano3", required=True)
    ap.add_argument("--mine", required=True)
    ap.add_argument("--ratio", type=float, required=True, help="share of training rows that are yours, (0, 1]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nano3-rows", type=int, default=None, help="cap Nano3 rows (evenly subsampled, order kept)")
    ap.add_argument("--max-repeats", type=int, default=1, help="allow up to this many passes over your tasks")
    ap.add_argument("--keep-solved", action="store_true", help="keep your tasks with pass_rate == 1.0")
    ap.add_argument("--drop-unsolved", action="store_true", help="drop your tasks with pass_rate == 0.0")
    ap.add_argument("--allow-historical", action="store_true")
    ap.add_argument("--val-mine", type=int, default=64, help="your rows held out for validation")
    ap.add_argument("--val-nano3", type=int, default=192, help="Nano3 rows held out for validation")
    ap.add_argument("--prompts-per-step", type=int, default=128, help="grpo.num_prompts_per_step, for the report")
    args = ap.parse_args()

    if not 0 < args.ratio <= 1:
        raise SystemExit("--ratio must be in (0, 1]")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- your rows: filter, sort easy -> hard, hold out validation --------------------
    mine_all = list(iter_jsonl(args.mine))
    for i, r in enumerate(mine_all):
        validate_row(r, f"{args.mine}:{i + 1}")
        r["pass_rate"] = pass_rate_of(r, args.allow_historical)
    dropped_solved = sum(1 for r in mine_all if r["pass_rate"] >= 1.0 and not args.keep_solved)
    dropped_unsolved = sum(1 for r in mine_all if r["pass_rate"] <= 0.0 and args.drop_unsolved)
    mine = [
        r for r in mine_all
        if not (r["pass_rate"] >= 1.0 and not args.keep_solved)
        and not (r["pass_rate"] <= 0.0 and args.drop_unsolved)
    ]
    mine.sort(key=lambda r: (-r["pass_rate"], str(r.get("task_id", ""))))
    val_idx = set(evenly_spaced(len(mine), min(args.val_mine, len(mine) // 10)))
    mine_val = [r for i, r in enumerate(mine) if i in val_idx]
    mine_train = [r for i, r in enumerate(mine) if i not in val_idx]
    if not mine_train:
        raise SystemExit("no training rows of yours left after filtering")

    # ---- Nano3 rows: count, pick validation rows, decide how many to keep ------------
    n_nano_total = 0 if args.ratio == 1 else count_lines(args.nano3)
    nano_val_idx = set(evenly_spaced(n_nano_total, args.val_nano3)) if n_nano_total else set()
    n_nano_avail = n_nano_total - len(nano_val_idx)
    n_nano = min(n_nano_avail, args.nano3_rows) if args.nano3_rows else n_nano_avail

    warnings = []
    if args.ratio == 1:
        repeats = args.max_repeats
        n_mine = len(mine_train) * repeats
        n_nano = 0
        warnings.append("ratio=1.0 trains on your data only. The tech report found single-environment RL "
                        "can degrade other skills; consider keeping a Nano3 replay slice (ratio < 1).")
    else:
        wanted = round(args.ratio / (1 - args.ratio) * n_nano)
        repeats = max(1, min(args.max_repeats, math.ceil(wanted / len(mine_train))))
        n_mine = min(wanted, len(mine_train) * repeats)
        if n_mine < wanted:  # not enough of yours: shrink Nano3 to hit the ratio
            n_nano = math.floor(n_mine * (1 - args.ratio) / args.ratio)
            warnings.append(f"only {n_mine} rows of yours available; Nano3 reduced to {n_nano} rows to keep "
                            f"ratio {args.ratio}. Raise --max-repeats or add tasks to keep more Nano3 data.")
        if len(mine_train) > wanted:
            n_mine = len(mine_train)
            warnings.append(f"you have more rows ({n_mine}) than the ratio needs ({wanted}); all kept, so the "
                            "actual ratio is higher than requested.")

    # Your rows, repeated as whole curriculum passes, trimmed to n_mine.
    mine_seq = [r for rep in range(repeats) for r in mine_train][:n_mine]

    # ---- proportional merge (both sides keep their order) ----------------------------
    def nano_stream() -> Iterator[dict[str, Any]]:
        if n_nano == 0:
            return
        mask = keep_mask(n_nano_avail, n_nano)
        for i, row in enumerate(iter_jsonl(args.nano3)):
            if i in nano_val_idx:
                continue
            if next(mask):
                yield row

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    nano_iter = nano_stream()
    k = j = 0
    next_nano = next(nano_iter, None)
    with open(train_path, "w") as out:
        while j < n_mine or next_nano is not None:
            mine_pos = (j + 0.5) / n_mine if j < n_mine else math.inf
            nano_pos = (k + 0.5) / n_nano if next_nano is not None else math.inf
            if mine_pos <= nano_pos:
                out.write(json.dumps(mine_seq[j]) + "\n")
                j += 1
            else:
                out.write(json.dumps(next_nano) + "\n")
                k += 1
                next_nano = next(nano_iter, None)

    with open(val_path, "w") as out:
        if nano_val_idx:
            for i, row in enumerate(iter_jsonl(args.nano3)):
                if i in nano_val_idx:
                    out.write(json.dumps(row) + "\n")
        for r in mine_val:
            out.write(json.dumps(r) + "\n")

    actual = j / (j + k) if (j + k) else 0
    per_step = args.prompts_per_step * actual
    if per_step < 8:
        warnings.append(f"only ~{per_step:.1f} of your prompts per GRPO step; your environment may get starved "
                        "in some batches. Raise --ratio or grpo.num_prompts_per_step.")
    report = {
        "train_rows": j + k,
        "train_rows_mine": j,
        "train_rows_nano3": k,
        "actual_ratio_mine": round(actual, 4),
        "your_prompts_per_step": round(per_step, 1),
        "passes_over_your_tasks": repeats,
        "val_rows_mine": len(mine_val),
        "val_rows_nano3": len(nano_val_idx),
        "dropped_mine_solved": dropped_solved,
        "dropped_mine_unsolved": dropped_unsolved,
        "approx_grpo_steps_one_epoch": math.ceil((j + k) / args.prompts_per_step),
        "warnings": warnings,
    }
    with open(os.path.join(args.out_dir, "blend_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
