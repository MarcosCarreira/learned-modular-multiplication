"""V3 — swing at H>90=3: richer Horner + Abacus embeddings + more capacity.

PyCharm Run. Holdout mode to measure tier-3 generalization (the H>90=3 test).
Bigger/longer than v2; the cleanly-separated reduction step (t_i mod p) + Abacus
digit alignment target the last 0.1 of tier 3. Resume-capable; ~10h cap.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS      = ["tier1", "tier2", "tier3"]
D_MODEL    = 384
HEADS      = 8
LAYERS     = 8
STEPS      = 200000
MINUTES    = 600
LR         = 1e-4       # lower than v2's 3e-4 — the bigger model diverged at 3e-4
WARMUP     = 2000       # linear LR warmup to stabilize the 20M-param start
CURRICULUM = True
CURR_FRAC  = 0.2
PRINT_SECS = 60
EVAL_SECS  = 1500
EVAL_N     = 300        # smaller eval batch — the big model's generation is memory-heavy
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v3"
OUT       = "models/recurrent_v3/weights.pt"
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out = str((REPO / OUT).resolve())
argv = ["train_recurrent.py", "--bands", *BANDS, "--holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--d-model", str(D_MODEL), "--heads", str(HEADS), "--layers", str(LAYERS),
        "--steps", str(STEPS), "--lr", str(LR), "--warmup", str(WARMUP),
        "--eval-secs", str(EVAL_SECS),
        "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N), "--device", DEVICE, "--out", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", str(CURR_FRAC)]
if (REPO / OUT).exists():
    argv += ["--resume", "--init", out]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
