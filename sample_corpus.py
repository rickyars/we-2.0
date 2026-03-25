"""
Sample and analyze the confession corpus.

Two modes:
  --source csv    Read raw CSV (title + selftext), apply combine_text logic before analysis
  --source shards Read processed parquet shards from prepare.py cache (default, most accurate)

Usage:
    # Analyze what actually goes into training (after prepare.py has run)
    python sample_corpus.py --source shards

    # Analyze short posts in the processed training data
    python sample_corpus.py --source shards --short

    # Analyze raw CSV before prepare.py
    python sample_corpus.py --source csv --csv "E:/projects/we-2.0/training-data/combined-confessions.csv"

    # Sample by subreddit (csv mode only, shards don't have subreddit column)
    python sample_corpus.py --source csv --csv "..." --subreddit confession --n 10

    # Save to file
    python sample_corpus.py --source shards --short --output short-analysis.txt
"""

import os
import argparse
import pandas as pd
import pyarrow.parquet as pq

# Mirror the logic from prepare.py
REMOVED_STRINGS = {"[removed]", "[deleted]"}
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
VAL_FILENAME = "shard_val.parquet"

def combine_text(title, selftext):
    title = (title or "").strip()
    selftext = (selftext or "").strip()
    if selftext in REMOVED_STRINGS:
        return title
    parts = [p for p in [title, selftext] if p]
    return "\n".join(parts)

def load_shards(max_rows=100_000):
    """Load text from parquet shards (training split only)."""
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
            if len(texts) >= max_rows:
                return texts[:max_rows]
    return texts

def load_csv_combined(csv_path, max_rows=None):
    """Load CSV and apply combine_text logic, return list of combined strings."""
    df = pd.read_csv(csv_path, usecols=["title", "selftext"])
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    df["text"] = df.apply(
        lambda row: combine_text(row.get("title"), row.get("selftext")),
        axis=1
    )
    return df

def main():
    parser = argparse.ArgumentParser(description="Sample and analyze confession corpus")
    parser.add_argument("--source", choices=["shards", "csv"], default="shards",
                        help="Where to read from: processed parquet shards or raw CSV (default: shards)")
    parser.add_argument("--csv", default=None, help="Path to raw CSV (required for --source csv)")
    parser.add_argument("--n", type=int, default=5, help="Number of samples (default: 5)")
    parser.add_argument("--subreddit", default=None, help="Filter by subreddit (csv mode only)")
    parser.add_argument("--short", action="store_true", help="Analyze short posts instead of sampling")
    parser.add_argument("--short-threshold", type=int, default=50, help="Char threshold for --short (default: 50)")
    parser.add_argument("--output", default=None, help="Save output to file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = []

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    if args.source == "shards":
        print("Loading from parquet shards...")
        texts = load_shards(max_rows=200_000)
        df = pd.DataFrame({"text": texts})
        has_subreddit = False

    else:  # csv
        if not args.csv:
            print("--csv is required when using --source csv")
            return
        print(f"Loading CSV and applying combine_text logic...")
        df = load_csv_combined(args.csv)
        has_subreddit = "subreddit" in pd.read_csv(args.csv, nrows=1).columns
        if has_subreddit:
            meta = pd.read_csv(args.csv, usecols=["subreddit"])
            df["subreddit"] = meta["subreddit"]

    total = len(df)
    lines.append(f"Total posts loaded: {total:,}")

    # ---------------------------------------------------------------------------
    # Short post analysis
    # ---------------------------------------------------------------------------
    if args.short:
        threshold = args.short_threshold
        short = df[df["text"].str.len() < threshold]

        lines.append(f"Short post analysis (combined text < {threshold} chars)")
        lines.append(f"Posts under {threshold} chars: {len(short):,} ({len(short)/total*100:.1f}%)")

        if has_subreddit and "subreddit" in df.columns:
            subreddits = sorted(short["subreddit"].dropna().unique())
        else:
            subreddits = ["all"]

        for sub in subreddits:
            if sub == "all":
                sub_short = short
            else:
                sub_short = short[short["subreddit"] == sub]

            lines.append(f"\n{'='*60}")
            lines.append(f"r/{sub} — {len(sub_short):,} posts under {threshold} chars")
            lines.append(f"{'='*60}")

            samples = sub_short.sample(min(args.n, len(sub_short)), random_state=args.seed)
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                lines.append(f"\n--- Sample {i} ({len(row['text'])} chars) ---")
                lines.append(repr(row["text"]))

    # ---------------------------------------------------------------------------
    # Standard sampling
    # ---------------------------------------------------------------------------
    else:
        if has_subreddit and "subreddit" in df.columns:
            lines.append(f"\n--- Subreddit breakdown ---")
            counts = df["subreddit"].value_counts()
            for sub, count in counts.items():
                pct = count / total * 100
                lines.append(f"  r/{sub:<25} {count:>8,}  ({pct:.1f}%)")

        if args.subreddit and has_subreddit:
            sample_df = df[df["subreddit"] == args.subreddit]
        else:
            sample_df = df

        samples = sample_df.sample(min(args.n, len(sample_df)), random_state=args.seed)

        lines.append(f"\n{'='*60}")
        lines.append(f"{'='*60}")
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            lines.append(f"\n--- Sample {i} ({len(row['text'])} chars) ---")
            text = row["text"]
            lines.append(text[:800] + ("...[truncated]" if len(text) > 800 else ""))

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    output = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()