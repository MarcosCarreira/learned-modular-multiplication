"""Auto-chunked v6 driver — the 20M / 14-layer tier-5 push, memory-safe.  Run and leave it.

On first launch, if the seed (models/recurrent_v6/weights.pt) is missing, it builds it
from v5 via run_build_v6_warmstart.py (8 layers copied + 6 near-identity on top). Then it
runs run_recurrent_v6.py as short FRESH subprocesses: each chunk exits (~40-min cap), the OS
reclaims its MPS memory, and the next resumes from the latest checkpoint (global step +
optimizer + *.metrics.csv saved every eval). Watch models/recurrent_v6/weights.pt.metrics.csv
— tier5_val is the number: >= 0.90 is H>90=5. Expect a long run (~1-2 weeks). Stops at
STEP_TARGET; re-run to continue. Keep this the ONLY MPS job running.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

# ----------------------------- CONFIG -----------------------------
STEP_TARGET = 33000    # must match STEPS in run_recurrent_v6.py (lowered to trigger the anneal)
MAX_CHUNKS  = 400      # safety ceiling (~40 min each); re-run to go further
PAUSE_S     = 4
# ------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_v6.py"
SEEDER = REPO / "run_build_v6_warmstart.py"
CKPT        = REPO / "models" / "recurrent_v6" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_v6" / "weights.pt.latest"


def ckpt_step() -> int:
    for p in (CKPT_LATEST, CKPT):
        if p.exists():
            try:
                return int(torch.load(p, map_location="cpu", weights_only=True).get("step", 0))
            except Exception as e:  # noqa: BLE001
                print(f"[auto] could not read {p.name} step ({e})")
    return 0


# 0) Seed the depth-warmstart from v5 if we have nothing yet.
if not CKPT.exists() and not CKPT_LATEST.exists():
    print("[auto] no v6 checkpoint — building the v5->v6 depth warm-start seed first ...", flush=True)
    r = subprocess.run([sys.executable, str(SEEDER)])
    if r.returncode != 0 or not CKPT.exists():
        print("[auto] warm-start seeding FAILED — see output above. Aborting.", flush=True)
        raise SystemExit(1)

prev_step, stuck = -1, 0
for chunk in range(1, MAX_CHUNKS + 1):
    s = ckpt_step()
    if s >= STEP_TARGET:
        print(f"[auto] checkpoint at step {s} >= target {STEP_TARGET} — done.", flush=True)
        break
    if s == prev_step:
        stuck += 1
        if stuck >= 2:
            print(f"[auto] no step progress across 2 chunks (stuck at {s}); aborting — "
                  f"check the worker output above for an error.", flush=True)
            break
    else:
        stuck = 0
    prev_step = s
    print(f"\n[auto] ===== chunk {chunk}/{MAX_CHUNKS} : resuming from step {s} "
          f"(fresh process, memory reclaimed) =====", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(WORKER)])
    print(f"[auto] chunk {chunk} exited code {r.returncode} after {(time.time()-t0)/60:.0f} min; "
          f"now at step {ckpt_step()}.", flush=True)
    time.sleep(PAUSE_S)
else:
    print(f"[auto] hit MAX_CHUNKS={MAX_CHUNKS}; stopping. Re-run to continue.", flush=True)

print(f"[auto] FINAL checkpoint step: {ckpt_step()}", flush=True)
print("[auto] Read models/recurrent_v6/weights.pt.metrics.csv — tier5_val is the number that matters.")
