"""SUBMISSION #5 candidate — no-holdout richer-Horner v3 (the H>90=3 swing).  PyCharm ▶.

Warm-starts from the converged HOLDOUT v3 (models/recurrent_v3/weights.pt — the run
that reached H>90=3, tier-3 ~0.92 on UNSEEN primes after the LR anneal) and trains
NO-HOLDOUT on all tier-1/2 primes + tier-3 sample, full 5d operands (no curriculum —
the model already handles 5d). Gentle LR annealed to zero to LOCK IN the tier-3 ≥0.90
while fitting every tier-2 table (lifts tier-2 from the holdout's ~0.70 toward ~0.97).

Target: tier-3 >= 0.90 on the official benchmark -> H>90 = 3, beating live #4 (H>90=2).
After it converges: run_check_submission.py (point SUBMISSION at this dir) to confirm
H>90=3 + overall_accuracy on the official pipeline, then run_upload_hf.py for #5.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3"]
BAND_WEIGHTS = [0.4, 1.0, 2.6]   # oversample tier3: don't burn capacity perfecting the already-won tier2
D_MODEL   = 384
HEADS     = 8
LAYERS    = 8
STEPS     = 40000       # gentle finetune horizon; LR cosine-anneals to 0 by here
MINUTES   = 75          # SHORT chunk: run_recurrent_v3_ship_auto.py relaunches fresh processes to reclaim MPS memory
LR        = 3e-5        # gentle — warm-started from a converged H>90=3 model
WARMUP    = 500
PRINT_SECS = 60
EVAL_SECS  = 1800       # less frequent evals -> fewer memory spikes
EVAL_N     = 200        # was 800; the 20M model's gen-eval is memory-heavy on MPS (n=800 spiked PyCharm to ~75GB)
DEVICE  = "auto"
WARMSTART_FROM = "models/recurrent_v3/weights.pt"
MODEL_DIR = "models/recurrent_v3_ship"
OUT       = "models/recurrent_v3_ship/weights.pt"
# ------------------------------------------------------------------
# NOTE: no --curriculum flag -> trainer uses full operand range [0,p) (true 5d).
# First launch warm-starts from the holdout v3 (init != out -> fresh schedule/best,
# correct for a new shipping context). Re-launches continue this dir (resume-aware).

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out  = str((REPO / OUT).resolve())
init = out if (REPO / OUT).exists() else str((REPO / WARMSTART_FROM).resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--no-holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--steps", str(STEPS), "--lr", str(LR), "--warmup", str(WARMUP),
        "--eval-secs", str(EVAL_SECS), "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", init]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
