"""
Extract Reddit posts from Watchful1 Pushshift .zst dump files.
No filtering — just extracts title, selftext, and subreddit.
All filtering is handled downstream in prepare.py.

Usage:
    python extract_torrent_v2.py --dumps-dir "E:/reddit-dumps" --output "E:/projects/we-2.0/training-data/combined-confessions.csv"

Install dependencies first:
    pip install zstandard pandas tqdm
"""

import os
import io
import json
import argparse
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Target subreddits
# ---------------------------------------------------------------------------

TARGET_SUBREDDITS = {
    "confession",
    "confessions",
    "offmychest",
    "TrueOffMyChest",
    "tifu",
    "AmItheAsshole",
}

# ---------------------------------------------------------------------------
# ZST reader — no filtering
# ---------------------------------------------------------------------------

def read_zst_submissions(filepath):
    """
    Yield (title, selftext, subreddit) tuples from a Pushshift .zst submissions file.
    Skips lines that fail JSON parsing or have neither title nor selftext.
    No other filtering.
    """
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = obj.get("title") or ""
                selftext = obj.get("selftext") or ""
                subreddit = obj.get("subreddit") or "unknown"
                # Only skip if both are empty — nothing to work with
                if not title.strip() and not selftext.strip():
                    continue
                yield title, selftext, subreddit

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Reddit corpus from Pushshift dumps")
    parser.add_argument("--dumps-dir", required=True, help="Directory containing Watchful1 .zst files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    all_rows = []

    dump_files = sorted(f for f in os.listdir(args.dumps_dir) if f.endswith(".zst"))
    if not dump_files:
        print(f"No .zst files found in {args.dumps_dir}")
        return

    for fname in dump_files:
        subreddit_name = fname.replace("_submissions.zst", "").replace(".zst", "")

        matched = None
        for target in TARGET_SUBREDDITS:
            if subreddit_name.lower() == target.lower():
                matched = target
                break

        if not matched:
            print(f"Skipping {fname}")
            continue

        filepath = os.path.join(args.dumps_dir, fname)
        print(f"\nReading r/{matched}: {fname}")

        rows = []
        try:
            for title, selftext, subreddit in tqdm(read_zst_submissions(filepath), desc=f"  r/{matched}", unit=" posts"):
                rows.append({
                    "title": title,
                    "selftext": selftext,
                    "subreddit": subreddit,
                })
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
            continue

        print(f"  {len(rows):,} posts from r/{matched}")
        all_rows.extend(rows)

    if not all_rows:
        print("No posts extracted. Check your dumps directory and file names.")
        return

    # Dedup on title+selftext combined, shuffle
    print(f"\nTotal before dedup: {len(all_rows):,}")
    df = pd.DataFrame(all_rows)
    df["_dedup_key"] = df["title"].fillna("") + "|||" + df["selftext"].fillna("")
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total after dedup:  {len(df):,}")

    # Summary
    print(f"\n--- Subreddit breakdown ---")
    counts = df["subreddit"].value_counts()
    total = len(df)
    for sub, count in counts.items():
        pct = count / total * 100
        print(f"  r/{sub:<25} {count:>8,}  ({pct:.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"Columns: title, selftext, subreddit")
    print(f"Next step: run uv run prepare.py --reset")

if __name__ == "__main__":
    main()