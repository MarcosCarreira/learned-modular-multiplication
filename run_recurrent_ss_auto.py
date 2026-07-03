"""Auto-chunked scheduled-sampling driver — the drift-reduction swing.  Run and leave it.

Launch ONLY after v6's anneal has finished, IF tier5 fell short of 0.90 and you want to attack
the rollout drift (exposure bias) instead of adding capacity. On first launch it seeds
models/recurrent_v7_ss/weights.pt from the annealed v6 checkpoint (verbatim; same 14-layer arch;
step reset to 0 for a fresh schedule), then runs run_recurrent_ss.py as short FRESH subprocesses
(reclaim MPS memory, resume from latest each chunk). Watch models/recurrent_v7_ss/weights.pt.metrics.csv
— tier5_val is the number. Anneal (shorten STEPS in run_recurrent_ss.py) once it plateaus.
"""
import subprocess
import sys
import time
from pathlib import Path

import torch

# ----------------------------- CONFIG -----------------------------
STEP_TARGET = 40000    # must match STEPS in run_recurrent_ss.py
MAX_CHUNKS  = 400
PAUSE_S     = 4
# ------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent
WORKER = REPO / "run_recurrent_ss.py"
V6_FINAL    = REPO / "models" / "recurrent_v6" / "weights.pt"
CKPT        = REPO / "models" / "recurrent_v7_ss" / "weights.pt"
CKPT_LATEST = REPO / "models" / "recurrent_v7_ss" / "weights.pt.latest"


def ckpt_step() -> int:
    for p in (CKPT_LATEST, CKPT):
        if p.exists():
            try:
                return int(torch.load(p, map_location="cpu", weights_only=True).get("step", 0))
            except Exception as e:  # noqa: BLE001
                print(f"[auto] could not read {p.name} step ({e})")
    return 0


# 0) Seed from v6's final checkpoint if we have nothing yet (verbatim; same arch; fresh schedule).
if not CKPT.exists() and not CKPT_LATEST.exists():
    if not V6_FINAL.exists():
        print(f"[auto] need v6's final checkpoint at {V6_FINAL} — let v6 finish first. Aborting.", flush=True)
        raise SystemExit(1)
    print("[auto] seeding v7_ss from the annealed v6 checkpoint (step reset to 0) ...", flush=True)
    ck = torch.load(V6_FINAL, map_location="cpu", weights_only=True)
    ck["step"] = 0
    ck["h90"] = -1
    ck["frontier_val"] = -1.0
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, CKPT)
    print(f"[auto] seeded -> {CKPT} (config {ck.get('config')})", flush=True)

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
print("[auto] Read models/recurrent_v7_ss/weights.pt.metrics.csv — tier5_val is the number.")
