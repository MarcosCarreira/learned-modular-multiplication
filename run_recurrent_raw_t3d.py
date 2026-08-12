"""RAW tier-3 push, STAGE 1 -- crank the drift lever, warm-start the best.  PyCharm run.

recurrent_raw_t3 plateaued at dev tier3 0.84 / OFFICIAL 0.69 (a big -0.15 dev->official
shave: the ~40-step two-phase rollout generalises worse to the secret primes than the dev
holdout showed). Train loss went to 0 while dev stayed flat -> capacity is saturated, the
ceiling is DRIFT, not fitting. So stage 1 pulls the one lever that attacks drift and hasn't
been pushed to the wall yet: STRONGER scheduled sampling (ss-max 0.30 -> 0.55, earlier ramp),
warm-started from the t3 best, with tier3 weighted heavier and a fresh cosine anneal.

Honest expectation: scheduled sampling is the APPROXIMATE drift fix (it exposes the model to
its own tokens but keeps ground-truth targets; it does NOT relabel from the carried acc). It
already gave 0.70->0.84 dev once; cranking it may buy a few more points but is unlikely alone
to clear 0.90 official (that needs ~0.93-0.95 dev given the shave). Read this run as: does
harder on-policy move the OFFICIAL number at all? If it plateaus again, the honest call is the
drift wall holds and stage 2 (true expert-relabel DAgger, or inference-time self-consistency
voting) is the only thing left. tier2 stays no-holdout so H>90=2 is protected throughout.
Launch via run_recurrent_raw_t3d_auto.py.
"""
import sys
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
BANDS        = ["tier1", "tier2", "tier3"]
BAND_WEIGHTS = [0.10, 0.40, 3.0]   # tier3 heavier still; tier2 maintenance (already 0.99 no-holdout)
FULL_TIERS   = ["tier1", "tier2"]  # keep tier2 no-holdout -> protects H>90=2 while we chase tier3
D_MODEL   = 384
HEADS     = 8
LAYERS    = 8
BATCH     = 32
STEPS     = 60000       # fresh cosine -> 0 by 60k (finishing anneal built in)
MINUTES   = 40
LR        = 4e-5        # warm fine-tune, a touch below t3's 5e-5 (protect the 0.84 start)
WARMUP    = 800
CURRICULUM = False      # full 20-digit raw operands from the start
RAW_OPERANDS = True
SCHED_SAMP = True       # THE lever -- pushed harder this run
SS_MAX    = 0.55        # up from 0.30: substitute the model's own tokens far more aggressively
SS_RAMP   = 0.40        # reach full substitution by 40% of steps, then hold (more on-policy time)
EMPTY_CACHE_EVERY = 1
PRINT_SECS = 60
EVAL_SECS  = 2400
EVAL_N     = 100
EVAL_SUB   = 8
EVAL_VAL_ONLY = True
EVAL_EVERY = 1000000
DEVICE  = "auto"
MODEL_DIR = "models/recurrent_raw_t3d"
OUT       = "models/recurrent_raw_t3d/weights.pt"
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
        *(["--raw-operands"] if RAW_OPERANDS else []),
        *(["--full-tiers", *FULL_TIERS] if FULL_TIERS else []),
        *(["--sched-samp", "--ss-max", str(SS_MAX), "--ss-ramp", str(SS_RAMP)] if SCHED_SAMP else []),
        "--band-weights", *[str(w) for w in BAND_WEIGHTS],
        "--device", DEVICE, "--out", out, "--resume", "--init", out]
if CURRICULUM:
    argv += ["--curriculum", "--curr-frac", "0.30"]
if MINUTES:
    argv += ["--minutes", str(MINUTES)]
sys.argv = argv
train_recurrent.main()
