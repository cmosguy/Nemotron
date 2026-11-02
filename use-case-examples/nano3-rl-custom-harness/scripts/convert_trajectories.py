#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn agent-harness trajectories into (1) SFT data and (2) RL task rows for NeMo Gym.

Your harness logs are proprietary, so this script works on a small canonical schema.
Edit `adapt_record()` once to map your log format onto it; everything else is generic.

Canonical trajectory (one JSON object per line):

    {
      "trajectory_id": "run-0001",          # unique per attempt
      "task_id": "ticket-4411",             # same value for repeated attempts at one task
      "tools": [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}],
      "messages": [                          # OpenAI chat format, in order
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "search_tickets", "arguments": "{\"query\": \"...\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "..."},
        {"role": "assistant", "content": "Done. Ticket 4411 is closed."}
      ],
      "outcome": {"success": true, "score": 1.0},   # what your harness/QA decided
      "verifier": {...}                     # whatever your reward function needs (answer key, expected state)
    }

Subcommands:

    sft  -> {"messages": [...], "tools": [...]}  rows for the Stage 1 SFT blend
            (stage1_sft/config/data_prep reads `messages_field: messages`, `tools_field: tools`).

    rl   -> NeMo Gym task rows: the prompt only (no assistant turns), tools in Responses API
            format, an `agent_ref` that routes the row to your agent server, and your verifier
            fields at the top level of the row. One row per task_id, not per attempt.

Examples:

    python convert_trajectories.py sft --input my_runs.jsonl --output my_sft.jsonl --min-score 1.0
    python convert_trajectories.py rl  --input my_runs.jsonl --output my_rl_tasks.jsonl \
        --agent-name my_harness_agent --dataset-name my_harness_v1
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from typing import Any, Iterable


# --------------------------------------------------------------------------------------
# 1. Map YOUR log format to the canonical schema. This is the only function you edit.
# --------------------------------------------------------------------------------------
def adapt_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return a canonical trajectory dict, or None to skip the record.

    The default assumes the input is already canonical. Replace the body with your
    mapping, for example:

        return {
            "trajectory_id": raw["run"]["id"],
            "task_id": raw["task"]["id"],
            "tools": [to_openai_tool(t) for t in raw["toolbox"]],
            "messages": [to_openai_message(step) for step in raw["steps"]],
            "outcome": {"success": raw["status"] == "resolved", "score": raw.get("qa_score", 0.0)},
            "verifier": {"expected_calls": raw["task"]["gold_actions"]},
        }
    """
    return raw


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warning: {path}:{n}: skipped invalid JSON ({e})", file=sys.stderr)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def score_of(traj: dict[str, Any]) -> float:
    out = traj.get("outcome") or {}
    if "score" in out and out["score"] is not None:
        return float(out["score"])
    return 1.0 if out.get("success") else 0.0


def prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything before the first assistant turn: the task as the policy first sees it."""
    prompt = []
    for m in messages:
        if m.get("role") == "assistant":
            break
        prompt.append({"role": m["role"], "content": m.get("content") or ""})
    return prompt


def task_key(traj: dict[str, Any]) -> str:
    if traj.get("task_id"):
        return str(traj["task_id"])
    digest = hashlib.sha256(json.dumps(prompt_messages(traj["messages"]), sort_keys=True).encode()).hexdigest()
    return f"prompt-{digest[:16]}"


def to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Chat Completions tool schema -> Responses API tool schema (the format Gym rows use)."""
    if "function" in tool:  # {"type": "function", "function": {...}}
        fn = tool["function"]
        return {
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            "strict": bool(fn.get("strict", False)),
        }
    return tool  # already flat Responses format


def check_trajectory(traj: dict[str, Any]) -> list[str]:
    problems = []
    msgs = traj.get("messages") or []
    if not msgs:
        problems.append("no messages")
        return problems
    if not any(m.get("role") == "user" for m in msgs):
        problems.append("no user message")
    if msgs[-1].get("role") != "assistant":
        problems.append("does not end with an assistant turn")
    open_calls = set()
    for m in msgs:
        for c in m.get("tool_calls") or []:
            open_calls.add(c.get("id"))
        if m.get("role") == "tool":
            open_calls.discard(m.get("tool_call_id"))
    if open_calls:
        problems.append(f"tool calls without results: {sorted(x for x in open_calls if x)}")
    return problems


def load_canonical(path: str) -> list[dict[str, Any]]:
    out = []
    for raw in read_jsonl(path):
        traj = adapt_record(raw)
        if traj is None:
            continue
        problems = check_trajectory(traj)
        if problems:
            print(f"warning: skipped {traj.get('trajectory_id', '?')}: {'; '.join(problems)}", file=sys.stderr)
            continue
        out.append(traj)
    return out


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------
def cmd_sft(args: argparse.Namespace) -> None:
    trajs = load_canonical(args.input)
    kept_per_task: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    rows = []
    for t in sorted(trajs, key=score_of, reverse=True):  # best attempts first
        if score_of(t) < args.min_score:
            continue
        key = task_key(t)
        if args.max_per_task and kept_per_task[key] >= args.max_per_task:
            continue
        fingerprint = hashlib.sha256(json.dumps(t["messages"], sort_keys=True).encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept_per_task[key] += 1
        rows.append({
            "messages": t["messages"],
            "tools": t.get("tools") or [],
            "metadata": {"trajectory_id": t.get("trajectory_id"), "task_id": key, "score": score_of(t)},
        })
    n = write_jsonl(args.output, rows)
    print(f"sft: {len(trajs)} valid trajectories -> {n} SFT rows "
          f"({len(kept_per_task)} distinct tasks, min_score={args.min_score}) -> {args.output}")


def cmd_rl(args: argparse.Namespace) -> None:
    trajs = load_canonical(args.input)
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for t in trajs:
        by_task[task_key(t)].append(t)

    rows = []
    for key, attempts in by_task.items():
        first = attempts[0]
        prompts = {json.dumps(prompt_messages(a["messages"]), sort_keys=True) for a in attempts}
        if len(prompts) > 1:
            print(f"warning: task {key} has {len(prompts)} different prompts across attempts; using the first",
                  file=sys.stderr)
        successes = sum(1 for a in attempts if score_of(a) >= args.success_threshold)
        row: dict[str, Any] = {
            "responses_create_params": {
                "input": prompt_messages(first["messages"]),
                "tools": [to_responses_tool(t) for t in (first.get("tools") or [])],
                "parallel_tool_calls": False,
            },
            "agent_ref": {"type": "responses_api_agents", "name": args.agent_name},
            "dataset": args.dataset_name,
            "task_id": key,
            # Success rate of your *old* agent on this task. A rough prior only: the curriculum
            # should use pass rates measured with your SFT checkpoint (see README, step 4).
            "historical_pass_rate": successes / len(attempts),
            "historical_attempts": len(attempts),
        }
        # Verifier fields go at the top level: Gym passes the row's fields to /verify.
        for k, v in (first.get("verifier") or {}).items():
            if k in row:
                raise SystemExit(f"verifier field {k!r} collides with a reserved row field")
            row[k] = v
        rows.append(row)

    n = write_jsonl(args.output, rows)
    print(f"rl: {len(trajs)} valid trajectories -> {n} task rows for agent {args.agent_name!r} -> {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sft", help="successful trajectories -> SFT messages rows")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--min-score", type=float, default=1.0, help="keep trajectories scoring at least this")
    s.add_argument("--max-per-task", type=int, default=4, help="cap attempts per task so easy tasks don't dominate")
    s.set_defaults(func=cmd_sft)

    r = sub.add_parser("rl", help="trajectories -> NeMo Gym task rows (prompt + tools + verifier fields)")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--agent-name", required=True, help="Gym agent server name from your agent config YAML")
    r.add_argument("--dataset-name", default="my_harness_tasks")
    r.add_argument("--success-threshold", type=float, default=1.0)
    r.set_defaults(func=cmd_rl)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
