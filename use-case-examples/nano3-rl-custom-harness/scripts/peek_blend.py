#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sample records from nvidia/Nemotron-3-Nano-RL-Training-Blend without downloading 6.9 GB.

The blend's train.jsonl is ordered by curriculum (easy -> hard). This script reads a few
small byte ranges spread across the file over HTTP, keeps the complete JSON lines in each
range, and prints a per-environment summary plus how the pass rate changes along the file.

Usage:
    python peek_blend.py --chunks 12 --chunk-mb 3 --out sampled.jsonl
    HF_TOKEN=... python peek_blend.py          # token only needed if you hit rate limits

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import urllib.request

URL = "https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend/resolve/main/train.jsonl"
FILE_SIZE = 6_924_505_168  # bytes, from the dataset's file listing


def fetch_range(start: int, length: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def complete_lines(blob: bytes, at_file_start: bool) -> list[dict]:
    lines = blob.split(b"\n")
    body = lines[:-1] if at_file_start else lines[1:-1]  # drop partial first/last line
    out = []
    for raw in body:
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=int, default=12, help="number of byte ranges spread across the file")
    ap.add_argument("--chunk-mb", type=float, default=3.0, help="size of each range in MB")
    ap.add_argument("--out", default=None, help="optional JSONL file to write the sampled records to")
    args = ap.parse_args()

    length = int(args.chunk_mb * 1_000_000)
    stride = (FILE_SIZE - length) // max(1, args.chunks - 1)
    records = []
    for i in range(args.chunks):
        start = i * stride
        recs = complete_lines(fetch_range(start, length), at_file_start=(start == 0))
        for r in recs:
            r["_offset_fraction"] = round(start / FILE_SIZE, 3)
        records.extend(recs)
        mean = sum(r.get("pass_rate", 0) for r in recs) / len(recs) if recs else float("nan")
        print(f"range {i:2d} @ {start / FILE_SIZE:5.1%} of file: {len(recs):4d} records, mean pass_rate {mean:.2f}")

    print("\nRecords per environment (agent_ref.name):")
    counts = collections.Counter(r.get("agent_ref", {}).get("name") for r in records)
    for name, n in counts.most_common():
        print(f"  {name:45s} {n}")

    placeholders = sum(1 for r in records if "_hf_placeholder" in r)
    print(f"\nPlaceholder rows (DAPO/Skywork, resolved by data_prep.py): {placeholders}")

    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
