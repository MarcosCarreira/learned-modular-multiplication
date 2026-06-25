"""V4 worker — the H>90=4 swing: wider operands + base-1000 Horner, HOLD-OUT.

Trains the recurrent_v4 scratchpad (WIDTH=10, base-1000 Horner, NB=4) on tiers 1-4
in HOLD-OUT mode, so tier-4 val measures genuine generalization to unseen ~10-digit
primes. Tier-4 is oversampled (band weights) so capacity goes where the bar is
contested; the length curriculum grows operands 1->10 digits.

Runs as ONE 75-min chunk and self-caps — launch it via run_recurrent_v4_auto.py,
which relaunches fresh subprocesses (reclaims MPS memory) and resumes seamlessly.
Goal calibration: tier-4 val ~0.6 is already a promising "the operation scales past
tier 3" signal; ~0.9 would be H>90=4.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3", "tier4"]
BAND_WEIGHTS = [0.3, 0.5, 1.0, 2.6]   # oversample tier4 (the contested tier)
D_MODEL   = 384         # capable size kept; memory pinned by per-step cache release + small batch
HEADS     = 8
LAYERS    = 8
BATCH     = 32          # ~31GB peak at seq 394 — confirmed manageable; keeps convergence snappy
STEPS     = 120000      # shortened from 200k: forces the cosine LR to anneal to 0 by ~120k (the finishing lift, ~3h from step 96k)
MINUTES   = 25          # SHORT chunk; the auto driver relaunches fresh processes to reclaim memory
LR        = 1e-4
WARMUP    = 2000
CURRICULUM = True
CURR_FRAC  = 0.25       # operand cap ramps 1->10 digits over 50k steps
EMPTY_CACHE_EVERY = 1   # release MPS cache EVERY step -> resident stays ~ one step's peak, no growth
PRINT_SECS = 60
EVAL_SECS  = 1800       # eval sub-batched (16 seqs/chunk in exact_match) + cache-cleared at the boundary
EVAL_N     = 200        # bigger sample = less noisy reads; memory unchanged (still generated 16 at a time)
EVAL_EVERY = 1000000    # disable step-based evals; eval only on the time schedule / chunk cap (rare)
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v4"
OUT       = "models/recurrent_v4/weights.pt"
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out = str((REPO / OUT).resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--batch", str(BATCH), "--empty-cache-every", str(EMPTY_CACHE_EVERY),
        "--steps", str(STEPS), "--lr", str(LR), "--warmup", str(WARMUP),
        "--eval-secs", str(EVAL_SECS), "--eval-every", str(EVAL_EVERY),
        "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
