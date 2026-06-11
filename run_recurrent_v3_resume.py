"""Resume v3 under the schedule-aware trainer — TRUE continuation.

Run this to continue an interrupted v3 session. It picks the curriculum + LR up
EXACTLY where session 1 stopped (no 1d re-ramp, no LR re-warmup) — same schedule
shape as run_recurrent_v3.py, just offset to the resumed step. If session 1 was at
3d, this keeps ramping 3d -> 4d -> 5d normally.

LEGACY STAMP: the first v3 session was trained before the resume fix, so its
weights.pt has no stored global step. Set RESUMED_STEP to the step on session 1's
LAST "saved best ... step N" line (that is the step the saved weights correspond
to — e.g. 15892 if that was the last best-save). Once a session has run under the
fixed trainer, the step is stored automatically and you set RESUMED_STEP = None.

FINISHING ANNEAL (optional, later): once this has climbed to 5d and tier-3 starts
to plateau, relaunch with STEPS lowered to ~60000 — that makes the LR anneal to
zero while dwelling at 5d, the regime that sharpened the v2 seeds into the 0.8s.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
RESUMED_STEP = None    # checkpoint now stamps its own step (it's at ~42000) — leave None
STEPS      = 60000     # ANNEAL PHASE: short horizon => LR cools to ~0 by step 60k while dwelling at 5d
MINUTES    = 360
LR         = 1e-4      # cosine over STEPS; ~2e-5 at the resumed step 42k, annealing to 0 at 60k
WARMUP     = 0         # already past warmup — no fresh warmup on resume
CURR_FRAC  = 0.2       # ramp end (12k) is far behind the resumed step => full 5d throughout
PRINT_SECS = 60
EVAL_SECS  = 1500
EVAL_N     = 300
DEVICE     = "auto"
MODEL_DIR  = "models/recurrent_v3"
OUT        = "models/recurrent_v3/weights.pt"
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
out_path = (REPO / OUT).resolve()

# Legacy stamp: write the global step into a pre-fix checkpoint so the schedule
# picks up at the right place. Runs on the Mac (has torch); harmless if already set.
if RESUMED_STEP is not None:
    import torch
    ck = torch.load(out_path, map_location="cpu", weights_only=True)
    if int(ck.get("step", 0)) == 0:
        ck["step"] = int(RESUMED_STEP)
        torch.save(ck, out_path)
        print(f"stamped step={RESUMED_STEP} into {out_path} (legacy checkpoint)")
    else:
        print(f"checkpoint already carries step={ck['step']}; ignoring RESUMED_STEP")

sys.path.insert(0, str(REPO / "dev"))
import train_recurrent  # noqa: E402

out = str(out_path)
argv = ["train_recurrent.py", "--bands", "tier1", "tier2", "tier3", "--holdout",
        "--model-dir", str((REPO / MODEL_DIR).resolve()),
        "--steps", str(STEPS), "--lr", str(LR), "--warmup", str(WARMUP),
        "--curriculum", "--curr-frac", str(CURR_FRAC),
        "--eval-secs", str(EVAL_SECS), "--print-secs", str(PRINT_SECS),
        "--eval-n", str(EVAL_N), "--device", DEVICE, "--out", out,
        "--resume", "--init", out]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
