"""SUBMISSION #6 candidate — no-holdout WIDTH-10 v4 (the H>90=4 swing).  PyCharm worker.

Warm-starts from the converged HOLDOUT v4 (models/recurrent_v4/weights.pt — which
cleared all four tiers on UNSEEN primes, tier-4 0.905, H>90=4 held out) and trains
NO-HOLDOUT on all tier-1/2/3 primes + a tier-4 sample, full 10-digit operands.
tier-4-weighted, gentle LR annealed to zero — fits every in-band prime while holding
the tier-4 generalisation, exactly as the v3 ship turned holdout 0.92 into official
0.93. Target: tier-4 >= 0.90 on the official benchmark -> H>90 = 4, beating live #5.

Runs as one ~25-min chunk and self-caps; launch via run_recurrent_v4_ship_auto.py.
After it converges: point run_check_submission.py at this dir to confirm H>90 + OA,
then run_upload_hf.py for #6.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3", "tier4"]
BAND_WEIGHTS = [0.3, 0.5, 1.0, 2.6]   # oversample tier4 (the contested tier)
D_MODEL   = 384
HEADS     = 8
LAYERS    = 8
BATCH     = 32          # seq ~394; keeps the MPS peak ~31GB
STEPS     = 60000       # gentle finetune horizon; LR cosine-anneals to 0 by here
MINUTES   = 25
LR        = 3e-5        # gentle — warm-started from a converged H>90=4 (holdout) model
WARMUP    = 500
EMPTY_CACHE_EVERY = 1   # per-step MPS release (bounds memory on the long seq)
PRINT_SECS = 60
EVAL_SECS  = 1800
EVAL_N     = 200        # sub-batched at 16 internally; memory bounded
DEVICE  = "auto"
WARMSTART_FROM = "models/recurrent_v4/weights.pt"
MODEL_DIR = "models/recurrent_v4_ship"
OUT       = "models/recurrent_v4_ship/weights.pt"
# ------------------------------------------------------------------
# NO --curriculum flag -> full operand range (the warm-start already handles 10d).

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out  = str((REPO / OUT).resolve())
init = out if (REPO / OUT).exists() else str((REPO / WARMSTART_FROM).resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--no-holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--batch", str(BATCH), "--empty-cache-every", str(EMPTY_CACHE_EVERY),
        "--steps", str(STEPS), "--lr", str(LR), "--warmup", str(WARMUP),
        "--eval-secs", str(EVAL_SECS), "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", init]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
