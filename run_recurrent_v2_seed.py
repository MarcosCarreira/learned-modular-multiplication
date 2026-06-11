"""2-SEED variance for the write-up.  PyCharm Run (no args).

Re-runs the holdout Horner v2 at a fresh seed to characterize variance on the
tier-3 generalization number (currently single-seed 0.79). Run once with SEED=1,
then change to SEED=2 and run again; read tier-3 val from each. Writes to its own
seed-specific weights file (does NOT touch the submission models).
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
SEED      = 2           # <-- run with 1, then change to 2 and run again
BANDS     = ["tier1", "tier2", "tier3"]
D_MODEL   = 256
HEADS     = 8
LAYERS    = 6
STEPS     = 150000
MINUTES   = 300         # ~5h to reach the tier-3 plateau (it converged ~step 60k)
CURRICULUM = True
CURR_FRAC  = 0.2
PRINT_SECS = 60
EVAL_SECS  = 1500
EVAL_N     = 600
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v2"            # reuse the arch
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out = str((REPO / f"models/recurrent_v2/weights_s{SEED}.pt").resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--holdout", "--seed", str(SEED),
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--steps", str(STEPS), "--eval-secs", str(EVAL_SECS),
        "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N), "--device", DEVICE, "--out", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
