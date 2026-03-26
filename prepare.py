"""
One-time data preparation for autoresearch experiments.
Converts local CSV corpus to parquet shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (convert + tokenizer)
    python prepare.py --num-shards 8   # convert only 8 training shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
import shutil
import json

import pandas as pd
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 10 * 524288  # was 40, scaled down 4x for MAX_SEQ_LEN=512

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "training-data", "watchful1-confessions.csv")
SHARD_SIZE = 10_000
VAL_SIZE = 10_000
VAL_FILENAME = "shard_val.parquet"
VOCAB_SIZE = 8192

# Minimum and maximum character length of combined text
MIN_TEXT_LENGTH = 40
MAX_TEXT_LENGTH = 2000

# Strings indicating deleted/removed content
REMOVED_STRINGS = {"[removed]", "[deleted]"}

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Text combination logic
# ---------------------------------------------------------------------------

def combine_text(title, selftext):
    """
    Combine title and selftext into a structured JSON training document.

    Format: {"title": "...", "confession": "..."}

    Rules:
    - If selftext is [removed] or [deleted]: confession field uses title only
    - Otherwise: title field gets the title, confession field gets selftext
    - If no title: title field is omitted
    """
    title = (title or "").strip()
    selftext = (selftext or "").strip()

    if selftext in REMOVED_STRINGS or not selftext:
        # Title-only post — confession IS the title
        if not title:
            return ""
        return json.dumps({"confession": title}, ensure_ascii=False)

    if title:
        return json.dumps({"title": title, "confession": selftext}, ensure_ascii=False)
    else:
        return json.dumps({"confession": selftext}, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Corpus conversion
# ---------------------------------------------------------------------------

def convert_corpus(num_shards=None):
    """Convert local CSV corpus to parquet shards + validation shard."""
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if os.path.exists(val_path) and len(list_parquet_files()) > 1:
        print(f"Data: shards already exist at {DATA_DIR}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Data: loading {CORPUS_PATH} ...")
    df = pd.read_csv(CORPUS_PATH, usecols=["title", "selftext"])

    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")

    # Combine title + selftext into structured JSON text
    df["text"] = df.apply(
        lambda row: combine_text(row.get("title"), row.get("selftext")),
        axis=1
    )

    # Filter: remove empty, too short, too long
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    df = df[df["text"].str.len() <= MAX_TEXT_LENGTH]
    df = df[["text"]]

    # Deduplicate and shuffle
    df = df.drop_duplicates(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Data: {len(df):,} rows after filtering")

    train_df = df.iloc[:-VAL_SIZE]
    val_df = df.iloc[-VAL_SIZE:]

    # Write validation shard
    val_df.to_parquet(val_path, index=False)
    print(f"Data: wrote validation shard ({len(val_df):,} rows) -> {val_path}")

    # Write training shards
    total_shards = math.ceil(len(train_df) / SHARD_SIZE)
    limit = min(num_shards, total_shards) if num_shards is not None else total_shards
    for i in range(limit):
        shard = train_df.iloc[i * SHARD_SIZE:(i + 1) * SHARD_SIZE]
        path = os.path.join(DATA_DIR, f"shard_{i + 1:05d}.parquet")
        shard.to_parquet(path, index=False)
    print(f"Data: wrote {limit} training shards to {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Run prepare.py to convert corpus first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    One-confession-per-row dataloader.
    Every row contains exactly one confession, padded with BOS tokens to fill T.
    This prevents cross-confession context bleed and keeps the model focused on
    generating coherent single documents.

    Row layout: [BOS] [confession tokens...] [BOS BOS BOS...] (padding)
    The leading BOS marks document start. Trailing BOS tokens fill unused space.
    The model learns to predict BOS after a confession ends, signaling completion.
    """
    assert split in ["train", "val"]
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers filled with BOS as padding
    row_buffer = torch.full((B, T + 1), bos_token, dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            # Refill buffer if needed
            while len(doc_buffer) < buffer_size:
                refill_buffer()

            # Pop one confession
            doc = doc_buffer.pop(0)

            # Reset row to BOS padding
            row_buffer[row_idx].fill_(bos_token)

            # Write confession into row, truncating if longer than T+1
            doc_len = min(len(doc), T + 1)
            row_buffer[row_idx, :doc_len] = torch.tensor(doc[:doc_len], dtype=torch.long)

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of training shards to convert (default: all). Val shard is always included.")
    parser.add_argument("--reset-data", action="store_true", help="Delete and re-convert data shards.")
    parser.add_argument("--reset-tokenizer", action="store_true", help="Delete and retrain tokenizer.")
    parser.add_argument("--reset", action="store_true", help="Reset both data and tokenizer (implies --reset-data --reset-tokenizer).")
    args = parser.parse_args()

    if args.reset or args.reset_data:
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            print(f"Removed {DATA_DIR}")
    if args.reset or args.reset_tokenizer:
        if os.path.exists(TOKENIZER_DIR):
            shutil.rmtree(TOKENIZER_DIR)
            print(f"Removed {TOKENIZER_DIR}")

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Convert corpus to parquet shards
    convert_corpus(num_shards=args.num_shards)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")