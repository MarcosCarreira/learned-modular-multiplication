"""SS worker — scheduled-sampling fine-tune to attack rollout DRIFT (the tier-5 wall).

The forum finding (Robby Sneiderman / neural-horner) crystallised what v6 also showed:
the long-scratchpad ceiling is a COVERAGE/DRIFT problem over the rollout, NOT a capacity
ceiling (a one-step overfit fits every operand family; the failure is compounding error
across the ~20 generated steps). More params (v6's depth) barely moved it. The right lever
is exposure-bias reduction.

This fine-tunes v6 with SCHEDULED SAMPLING: for a ramped fraction of completion positions the
model is fed its OWN predicted token instead of ground truth, so it learns to keep computing
correctly from its own (possibly drifted) context. Targets stay ground-truth. Costs a 2nd
forward pass per step.

  Compositional caveat: later scratchpad fields depend on earlier acc values, so fixed
  ground-truth targets under substitution is the standard-but-approximate form. If this helps
  but plateaus, the stronger version is DAgger with an EXPERT RELABEL -- generate the model's
  rollout, parse its carried acc at each step, and relabel the target to the arithmetically
  correct continuation FROM THAT acc (the expert is the trivial m=a*digit, t=acc*10+m, acc=t%p).
  That teaches true self-correction. Left as the follow-up if scheduled sampling underdelivers.

LAUNCH ONLY AFTER v6's anneal lands, IF tier5 fell short of 0.90 and you want the drift swing
instead of a bigger model. Warm-starts from models/recurrent_v6/weights.pt (via the auto driver).
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3", "tier4", "tier5"]
BAND_WEIGHTS = [0.3, 0.4, 0.7, 1.0, 2.0]   # tier5-leaning but diverse (diversity matters for drift)
D_MODEL   = 384
HEADS     = 8
LAYERS    = 14          # same arch as v6 (warm-started from it)
BATCH     = 12
STEPS     = 40000       # fine-tune horizon; shorten to anneal once tier5 plateaus
MINUTES   = 40
LR        = 2e-5        # gentle fine-tune (protect v6's weights while SS reshapes the rollout)
WARMUP    = 1000
SS_MAX    = 0.30        # max self-substitution prob
SS_RAMP   = 0.50        # ramp 0 -> SS_MAX over half the run
CURRICULUM = False
EMPTY_CACHE_EVERY = 1
PRINT_SECS = 60
EVAL_SECS  = 2400
EVAL_N     = 50
EVAL_SUB   = 4
EVAL_VAL_ONLY = True
EVAL_EVERY = 1000000
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_v7_ss"    # its own dir (submission-ready); same 14-layer arch as v6
OUT       = "models/recurrent_v7_ss/weights.pt"
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
        "--sched-samp", "--ss-max", str(SS_MAX), "--ss-ramp", str(SS_RAMP),
        "--eval-secs", str(EVAL_SECS), "--eval-every", str(EVAL_EVERY),
        "--print-secs", str(PRINT_SECS), "--eval-n", str(EVAL_N), "--eval-sub", str(EVAL_SUB),
        *(["--eval-val-only"] if EVAL_VAL_ONLY else []),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", "0.30"]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
