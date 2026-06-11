"""TIER-3 ATTEMPT — interleaved multiply-reduce (Horner) scratchpad.  PyCharm Run.

Never forms the full product; computes a*b mod p in 5 small Horner steps, so the
multiplication that stalled recurrent_v1 on tier3 is decomposed into per-digit
ops. Holdout mode + length curriculum. Writes to models/recurrent_v2/.
WATCH tier3 TRAIN (does the algorithm learn in-sample) then tier3 VAL (does it
generalize to unseen primes -- the prize). Resume-capable (re-run to continue).
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS     = ["tier1", "tier2", "tier3"]
D_MODEL   = 256
HEADS     = 8
LAYERS    = 6
STEPS     = 150000
MINUTES   = 600         # overnight; resume-capable, Ctrl-C anytime
CURRICULUM = True       # ease operand magnitude up over training
CURR_FRAC  = 0.2        # reach FULL operand size by ~30k steps (was 0.5 = too slow)
EVAL_EVERY = 100000     # high; the time triggers below govern cadence
PRINT_SECS = 60         # cheap heartbeat (step/loss/curr/throughput) every ~1 min
EVAL_SECS  = 1500       # full accuracy eval + save every ~25 min (the slow part)
EVAL_N     = 600        # samples per (now-rare) accuracy eval
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v2"
OUT       = "models/recurrent_v2/weights.pt"
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out = str((REPO / OUT).resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--steps", str(STEPS), "--eval-every", str(EVAL_EVERY),
        "--eval-secs", str(EVAL_SECS), "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N),
        "--device", DEVICE, "--out", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if (REPO / OUT).exists():
    argv += ["--resume", "--init", out]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
