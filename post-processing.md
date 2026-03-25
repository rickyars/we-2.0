# CONFESSOR.EXE — Training

The autoresearch loop found the best architecture.
Now you train it properly — not 5 minutes, but hours.
Read this fully before touching anything.

---

## Step 1 — Find the winning commit

Read `results.tsv`. Find the row with the lowest `val_bpb` and status `keep`.
That commit hash is your winner.

```powershell
Get-Content results.tsv
```

---

## Step 2 — Recover the winning train.py

```powershell
git show <winning_commit>:train.py > train_winner.py
```

---

## Step 3 — Set the training budget

Open `train_winner.py`. The `TIME_BUDGET` is imported from `prepare.py`
and fixed at 300 seconds. Override it by adding this line immediately
after the imports:

```python
TIME_BUDGET = 2*60*60  # 2 hours per session — adjust based on patience
```

This is the budget **per session**. Run multiple sessions to accumulate more
training. Each session resumes from the previous checkpoint.

| TIME_BUDGET | Expected quality |
|---|---|
| 300s (5 min) | Baseline — rough |
| 3600s (1 hr) | Noticeably better |
| 7200s (2 hrs) | Good — recommended starting point |
| 14400s (4 hrs) | Very good |
| 28800s (8 hrs) | Diminishing returns at this model size |

---

## Step 4 — Add checkpoint resume

At the start of the training script, after `optimizer = model.setup_optimizer(...)` and
before `model = torch.compile(model)`, add this block:

```python
# Checkpoint resume
CHECKPOINT_PATH = 'confession_model.pt'
checkpoint_step = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    checkpoint_step = ckpt.get('step', 0)
    print(f"Resumed from step {checkpoint_step}, running {TIME_BUDGET}s more")
```

Then find the line `step = 0` in the training loop setup and change it to:

```python
step = checkpoint_step
```

The LR schedule restarts fresh each session (warmup → peak → warmdown).
The optimizer state (momentum buffers) carries over — no warm-up penalty.

---

## Step 5 — Add checkpoint saving

At the very end of `train_winner.py`, after the final print block:

```python
# Save checkpoint (includes optimizer state for resumption)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': asdict(model.config),
    'step': step,
    'total_tokens': total_tokens,
}, CHECKPOINT_PATH)
print(f"Saved {CHECKPOINT_PATH}")
```

**Why optimizer state matters**: The optimizer (Muon + AdamW) accumulates
momentum buffers over training. Without them, resuming restarts optimization
from zero — the first hundred steps waste time recovering lost momentum instead
of making progress. Saving `optimizer_state_dict` prevents this.

---

## Step 6 — Run training (first session)

```powershell
uv run train_winner.py
```

Leave it. Come back in 2 hours.
`confession_model.pt` will be in the repo root.

---

### If training crashes

Likely OOM. Reduce `DEVICE_BATCH_SIZE` from 16 to 8 and retry.

---

## Step 7 — Test inference (between sessions)

Before running more sessions, verify the model (Steps 8–10 below).

If the model sounds coherent, run another session:

```powershell
uv run train_winner.py
```

The script auto-detects `confession_model.pt` and resumes. Run as many sessions as
you like — each adds 2 more hours of training with full optimizer continuity.

---

## Step 8 — Export to ONNX

Write `export_onnx.py` that:

1. Loads `confession_model.pt`
2. Reconstructs the GPT model from the saved config
3. Exports to ONNX with opset 17
4. Uses dynamic axes for batch size and sequence length
5. Saves as `confession_model.onnx`
6. Prints the file size in MB

Run it:
```powershell
uv run export_onnx.py
```

---

## Step 9 — Export the vocab

Write `export_vocab.py` that:

1. Loads the tokenizer from `~/.cache/autoresearch/tokenizer/tokenizer.pkl`
2. Exports the full vocabulary as `vocab.json` —
   a JSON object mapping token strings to integer IDs
3. Also exports the BOS token ID as a separate field

Run it:
```powershell
uv run export_vocab.py
```

---

## Step 10 — Test inference

Write `inference_test.py` that:

1. Loads `confession_model.onnx` with onnxruntime
2. Loads `vocab.json` for tokenization
3. Prompts with an empty BOS token (no text — cold generation)
4. Runs autoregressive generation for 300 tokens
5. Prints the result

Read it. Does it sound like a confession?
