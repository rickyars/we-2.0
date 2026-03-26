"""
Analyze confession size distributions in the processed parquet shards.
Helps understand how confessions are being packed into context windows.

Usage:
    python analyze_parquet.py
    python analyze_parquet.py --max-rows 100000
    python analyze_parquet.py --output parquet-analysis.txt
"""

import os
import math
import argparse
import pyarrow.parquet as pq
import pickle
import numpy as np

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_FILENAME = "shard_val.parquet"

MAX_SEQ_LEN = 2048  # must match prepare.py

def load_tokenizer():
    path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    with open(path, "rb") as f:
        import pickle
        return pickle.load(f)

def load_texts(max_rows=None):
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet") and f != VAL_FILENAME
    )
    texts = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts.extend(rg.column("text").to_pylist())
            if max_rows and len(texts) >= max_rows:
                return texts[:max_rows]
    return texts

def main():
    parser = argparse.ArgumentParser(description="Analyze parquet confession size distributions")
    parser.add_argument("--max-rows", type=int, default=50000, help="Max confessions to sample (default: 50000)")
    parser.add_argument("--output", default=None, help="Save output to file")
    args = parser.parse_args()

    print("Loading tokenizer...")
    enc = load_tokenizer()

    print(f"Loading up to {args.max_rows:,} confessions from parquet shards...")
    texts = load_texts(args.max_rows)
    print(f"Loaded {len(texts):,} confessions")

    print("Tokenizing...")
    lengths_chars = []
    lengths_tokens = []
    for text in texts:
        lengths_chars.append(len(text))
        lengths_tokens.append(len(enc.encode_ordinary(text)))

    lengths_chars = np.array(lengths_chars)
    lengths_tokens = np.array(lengths_tokens)

    lines = []

    # ---------------------------------------------------------------------------
    # Character distribution
    # ---------------------------------------------------------------------------
    lines.append("=" * 60)
    lines.append("CHARACTER LENGTH DISTRIBUTION")
    lines.append("=" * 60)
    lines.append(f"Count:   {len(lengths_chars):,}")
    lines.append(f"Min:     {lengths_chars.min():,}")
    lines.append(f"Mean:    {lengths_chars.mean():.0f}")
    lines.append(f"Median:  {np.median(lengths_chars):.0f}")
    lines.append(f"p75:     {np.percentile(lengths_chars, 75):.0f}")
    lines.append(f"p90:     {np.percentile(lengths_chars, 90):.0f}")
    lines.append(f"p95:     {np.percentile(lengths_chars, 95):.0f}")
    lines.append(f"p99:     {np.percentile(lengths_chars, 99):.0f}")
    lines.append(f"Max:     {lengths_chars.max():,}")

    # ---------------------------------------------------------------------------
    # Token distribution
    # ---------------------------------------------------------------------------
    lines.append("")
    lines.append("=" * 60)
    lines.append("TOKEN LENGTH DISTRIBUTION")
    lines.append("=" * 60)
    lines.append(f"Count:   {len(lengths_tokens):,}")
    lines.append(f"Min:     {lengths_tokens.min():,}")
    lines.append(f"Mean:    {lengths_tokens.mean():.0f}")
    lines.append(f"Median:  {np.median(lengths_tokens):.0f}")
    lines.append(f"p75:     {np.percentile(lengths_tokens, 75):.0f}")
    lines.append(f"p90:     {np.percentile(lengths_tokens, 90):.0f}")
    lines.append(f"p95:     {np.percentile(lengths_tokens, 95):.0f}")
    lines.append(f"p99:     {np.percentile(lengths_tokens, 99):.0f}")
    lines.append(f"Max:     {lengths_tokens.max():,}")

    # ---------------------------------------------------------------------------
    # Fit within context lengths
    # ---------------------------------------------------------------------------
    lines.append("")
    lines.append("=" * 60)
    lines.append("% CONFESSIONS FITTING WITHIN CONTEXT LENGTHS (tokens)")
    lines.append("=" * 60)
    for ctx in [64, 128, 256, 512, 1024, MAX_SEQ_LEN]:
        pct = (lengths_tokens <= ctx).mean() * 100
        lines.append(f"  <= {ctx:5d} tokens: {pct:5.1f}%")

    # ---------------------------------------------------------------------------
    # Packing simulation
    # ---------------------------------------------------------------------------
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"PACKING SIMULATION (MAX_SEQ_LEN={MAX_SEQ_LEN})")
    lines.append("=" * 60)
    lines.append("How many confessions end up packed per context window?")
    lines.append("")

    # Simulate greedy best-fit packing
    # Shuffle tokens lengths to simulate random ordering
    rng = np.random.default_rng(42)
    shuffled = lengths_tokens.copy()
    rng.shuffle(shuffled)

    confessions_per_row = []
    i = 0
    total = len(shuffled)
    while i < total:
        remaining = MAX_SEQ_LEN
        count = 0
        j = i
        # Simple sequential packing (approximate — real dataloader uses best-fit)
        while j < total and shuffled[j] <= remaining:
            remaining -= shuffled[j]
            count += 1
            j += 1
        if count == 0:
            # Single confession longer than MAX_SEQ_LEN — gets cropped
            count = 1
            j = i + 1
        confessions_per_row.append(count)
        i = j

    confessions_per_row = np.array(confessions_per_row)
    lines.append(f"Simulated rows:          {len(confessions_per_row):,}")
    lines.append(f"Mean confessions/row:    {confessions_per_row.mean():.1f}")
    lines.append(f"Median confessions/row:  {np.median(confessions_per_row):.0f}")
    lines.append(f"Min confessions/row:     {confessions_per_row.min()}")
    lines.append(f"Max confessions/row:     {confessions_per_row.max()}")
    lines.append("")

    # Distribution of confessions per row
    lines.append("Distribution of confessions per context window:")
    unique, counts = np.unique(confessions_per_row, return_counts=True)
    for val, count in zip(unique, counts):
        pct = count / len(confessions_per_row) * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {val:3d} confessions: {pct:5.1f}%  {bar}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("IMPLICATIONS")
    lines.append("=" * 60)
    single = (confessions_per_row == 1).mean() * 100
    lines.append(f"Rows with exactly 1 confession: {single:.1f}%")
    lines.append(f"Rows with 2+ confessions:       {100-single:.1f}%")
    lines.append("")
    lines.append("If you want one confession per context:")
    lines.append(f"  -> Filter to confessions >= {MAX_SEQ_LEN} tokens, or")
    lines.append(f"  -> Accept the packing and rely on BOS tokens as separators, or")
    lines.append(f"  -> Reduce MAX_SEQ_LEN to better match median confession length")
    median_tokens = int(np.median(lengths_tokens))
    lines.append(f"  -> Median confession is {median_tokens} tokens; MAX_SEQ_LEN={median_tokens*2} would give ~2 confessions/row on average")

    output = "\n".join(lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()