"""Score models through the OFFICIAL pipeline and print a per-tier table.
PyCharm ▶, no args.

MODELS entries are either a baseline name under examples/ or a path to a
submission dir. A submission needs its weights.pt to exist (train it first).
Reports per-tier accuracy, overall_accuracy, and H>90 (the primary key).
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
MODELS = [
    "dlp_grokking",                # strongest reference baseline
    "models/cross_prime_v1",       # your Track-1 model
    # "models/recip_v1",           # uncomment once Track-2 is trained
]
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import eval_all  # noqa: E402

# Submission paths (contain "/") -> absolute, so the PyCharm working dir doesn't
# matter. Bare names (e.g. "dlp_grokking") stay as examples/ names.
resolved = [str((REPO / m).resolve()) if "/" in m else m for m in MODELS]
sys.argv = ["eval_all.py", *resolved]
eval_all.main()
