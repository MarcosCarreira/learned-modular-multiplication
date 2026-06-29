"""V5 worker — the tier-5 scaling test: v4's recipe at WIDTH=20, HOLD-OUT.

Trains the recurrent_v5 scratchpad (WIDTH=20, base-10 a*single-digit Horner, NB=20)
on tiers 1-5 in HOLD-OUT mode, so tier-5 val measures genuine generalization to
unseen ~20-digit primes. Warm-started from v4 (the per-step op is identical; only
the length grows). Tier-5 is oversampled (band weights) so capacity goes where the
bar is contested; the length curriculum grows operands 1->20 digits.

Runs as ONE ~25-min chunk and self-caps — launch via run_recurrent_v5_auto.py, which
seeds the warm-start if missing and relaunches fresh subprocesses (reclaims MPS
memory) and resumes seamlessly. Calibration: tier-5 val ~0.6 already says "the
operation scales past tier 4"; ~0.9 would be H>90=5. Clearing 0.90 over 20 exact
steps is genuinely uncertain — that is the measurement.

MEMORY: seq 1384 is ~3.5x v4's, so BATCH is dropped to 16 (est. ~20GB resident,
under v4's ~31GB). If resident creeps over ~31GB, lower BATCH to 12 or 8; if it sits
well under, raise it for faster convergence. EMPTY_CACHE_EVERY=1 holds growth flat.
ANNEAL: like v4, once tier-5 val plateaus, SHORTEN STEPS to force the cosine LR to
zero around that point for the finishing lift, then let it run out.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3", "tier4", "tier5"]
BAND_WEIGHTS = [0.2, 0.3, 0.5, 0.8, 3.0]   # oversample tier5 (the contested tier)
D_MODEL   = 384         # same capacity as v4; memory pinned by per-step cache release + small batch
HEADS     = 8
LAYERS    = 8
BATCH     = 16          # seq 1384 (~3.5x v4); est. ~20GB resident, under v4's ~31GB peak
STEPS     = 75000       # ANNEAL TRIGGER (was 150k): curriculum done @20d + tiers lifted (t4~0.74/t5~0.54) -> drive cosine LR to 0 by 75k (~20k steps from ~55k) to consolidate near the peaks (v4: 0.76->0.91)
MINUTES   = 40          # longer chunk so the (slow) eval is amortized over more training
EVAL_VAL_ONLY = True    # val/held-out is the ranked metric; skip the train eval -> ~halves eval time
LR        = 1e-4
WARMUP    = 2000
CURRICULUM = True
CURR_FRAC  = 0.30       # operand cap ramps 1->20 digits over ~45k steps
EMPTY_CACHE_EVERY = 1   # release MPS cache EVERY step -> resident stays ~ one step's peak
PRINT_SECS = 60
EVAL_SECS  = 2400       # evals are slow at GEN_LEN=1320; one per chunk (at the 25-min cap) is enough
EVAL_N     = 60         # smaller eval sample (generation is ~3.5x longer than v4)
EVAL_SUB   = 4          # generate only 4 seqs at once during eval — bounds MPS memory at seq 1384
EVAL_EVERY = 1000000    # disable step-based evals; eval only on the chunk cap / time schedule
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v5"
OUT       = "models/recurrent_v5/weights.pt"
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
        "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N), "--eval-sub", str(EVAL_SUB),
        *(["--eval-val-only"] if EVAL_VAL_ONLY else []),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
