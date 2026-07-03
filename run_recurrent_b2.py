"""BASE-2 radix-ablation worker — quantify steps-vs-drift (base-2 vs base-10), NOT a submission.

Trains the base-2 (bit-serial) scratchpad from scratch on tiers 1-3 (holdout), with an
operand-magnitude curriculum. At tier 3 this is ~16 Horner steps vs base-10's ~5. The
comparison: read this model's tier3_val vs the base-10 recurrent_v3/v5 tier-3 (~0.90-0.95).
If base-2 trails, that's the steps-vs-drift evidence (more, smaller steps -> more rollout
drift), corroborating the v6 plateau and Sneiderman's base-2 diagnosis. Launch when the M1
is free (it competes with v6/SS for the machine). Argument-free: open in PyCharm, Run.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS     = ["tier1", "tier2", "tier3"]     # WIDTH=16 bits covers p < 2^16
D_MODEL   = 384
HEADS     = 8
LAYERS    = 8
BATCH     = 32          # seq 916 (< v5's 1384) -> batch 32 sits ~10-12GB
STEPS     = 80000
MINUTES   = 40
LR        = 3e-4        # from-scratch (higher LR than a warm fine-tune), cosine to 0
WARMUP    = 2000
CURRICULUM = True
CURR_FRAC  = 0.30       # operand-magnitude ramp (decimal cap; base-independent magnitude)
EMPTY_CACHE_EVERY = 1
PRINT_SECS = 60
EVAL_SECS  = 2400
EVAL_N     = 100
EVAL_SUB   = 8
EVAL_VAL_ONLY = True
EVAL_EVERY = 1000000
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_b2"
OUT       = "models/recurrent_b2/weights.pt"
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
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
