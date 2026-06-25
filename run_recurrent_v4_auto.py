"""Auto-chunked v4 driver — the H>90=4 swing, memory-safe.  PyCharm ▶ and leave it.

Runs run_recurrent_v4.py as a sequence of short FRESH subprocesses: each chunk
exits (75-min cap), the OS reclaims its MPS memory, and the next resumes exactly
where it stopped (trainer saves global step + optimizer + a *.metrics.csv). Watch
models/recurrent_v4/weights.pt.metrics.csv for the tier4 column — ~0.6 is the
promising signal, ~0.9 is H>90=4. Stops itself at STEP_TARGET; re-run to continue.
Keep this the ONLY MPS job running.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

# ----------------------------- CONFIG -----------------------------
STEP_TARGET = 120000   # must match STEPS in run_recurrent_v4.py (finishing-anneal horizon)
MAX_CHUNKS  = 48       # safety ceiling (~75 min each); re-run to go further
PAUSE_S     = 4
# ------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_v4.py"
CKPT        = REPO / "models" / "recurrent_v4" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_v4" / "weights.pt.latest"


def ckpt_step() -> int:
    for p in (CKPT_LATEST, CKPT):
        if p.exists():
            try:
                return int(torch.load(p, map_location="cpu", weights_only=True).get("step", 0))
            except Exception as e:  # noqa: BLE001
                print(f"[auto] could not read {p.name} step ({e})")
    return 0


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
print("[auto] Read models/recurrent_v4/weights.pt.metrics.csv — tier4_val is the number that matters.")
