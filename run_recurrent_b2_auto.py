"""Auto-chunked BASE-2 ablation driver.  Run and leave it.

Runs run_recurrent_b2.py as short FRESH subprocesses (reclaim MPS memory, resume from the
latest checkpoint each chunk). No warm-start seed — the base-2 model trains from scratch (its
vocab differs from base-10). Watch models/recurrent_b2/weights.pt.metrics.csv — compare its
tier3_val against base-10 recurrent_v3/v5 tier-3 (~0.90-0.95) once converged. Stops at
STEP_TARGET; re-run to continue. Keep this the ONLY MPS job running (it competes with v6/SS).
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

STEP_TARGET = 80000
MAX_CHUNKS  = 400
PAUSE_S     = 4

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_b2.py"
CKPT        = REPO / "models" / "recurrent_b2" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_b2" / "weights.pt.latest"


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
            print(f"[auto] no step progress across 2 chunks (stuck at {s}); aborting.", flush=True)
            break
    else:
        stuck = 0
    prev_step = s
    print(f"\n[auto] ===== chunk {chunk}/{MAX_CHUNKS} : resuming from step {s} =====", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(WORKER)])
    print(f"[auto] chunk {chunk} exited code {r.returncode} after {(time.time()-t0)/60:.0f} min; "
          f"now at step {ckpt_step()}.", flush=True)
    time.sleep(PAUSE_S)
else:
    print(f"[auto] hit MAX_CHUNKS={MAX_CHUNKS}; stopping. Re-run to continue.", flush=True)

print(f"[auto] FINAL checkpoint step: {ckpt_step()}", flush=True)
print("[auto] Compare models/recurrent_b2/weights.pt.metrics.csv tier3_val vs base-10 v3/v5 (~0.9+).")
