#!/usr/bin/env python3
"""
combine_datasets.py
Collect multiple Alpaca-style JSON files into a single train / test split.

Example
-------
python combine_datasets.py \
    --math  math_50k.json \
    --alpaca alpaca_data_cleaned.json \
    --commonsense commonsense_170k.json \
    --out_dir dataset/combined \
    --test_frac 0.05 \
    --seed 42
"""
import argparse, json, os, random
from pathlib import Path

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalise keys and fill in any missing fields
    for ex in data:
        ex.setdefault("input", "")
        ex.setdefault("answer", ex.get("output", ""))   # keeps evaluate.py happy
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--math", required=True)
    p.add_argument("--alpaca", required=True)
    p.add_argument("--commonsense", required=True)
    p.add_argument("--out_dir", default="dataset/combined")
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    # 1. Load & concatenate
    all_examples = (
          load(args.math)
        + load(args.alpaca)
        + load(args.commonsense)
    )

    # 2. Shuffle deterministically
    random.seed(args.seed)
    random.shuffle(all_examples)

    # 3. Train / test split
    # split_idx = int(len(all_examples) * (1 - args.test_frac))
    # train, test = all_examples[:split_idx], all_examples[split_idx:]

    # 4. Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)


    print(f"Combined {len(all_examples):,} examples — "
          f"{len(all_examples):,} train  "
          f"written to {out_dir}/")

if __name__ == "__main__":
    main()