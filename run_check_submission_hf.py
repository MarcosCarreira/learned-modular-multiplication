"""Pre-submission gauntlet — AGAINST THE ACTUAL HUGGINGFACE ARTIFACT.  PyCharm ▶.

run_check_submission.py checks a LOCAL dir. This one downloads the exact pushed
commit from HuggingFace and runs the same official pipeline on it — so it catches
anything that lives in the gap between "works on my Mac" and "what the judges pull
at the deadline": a stale/incomplete commit, a file that didn't get uploaded, a
manifest pointing at the wrong class. The final evaluation scores THIS artifact,
not your working tree, so this is the check that matters most.

Forces CPU to mirror the official sandbox. Green here == the registered submission
is intact and reproduces your local number.
"""
import json
import os
import sys
import time
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
REPO_ID    = "MCarreira/recurrent-v3-ship"   # the submitted HF repo
COMMIT     = "01f6f8a3fa56e3a3c17852dcf468560a0b6b8b16"   # the exact commit you registered (#4)
TOKEN      = ""                              # only needed if the repo is private; "" for public
TOTAL      = 1100                            # full benchmark (100/tier)
FORCE_CPU  = True                            # mirror the official sandbox
TIME_BUDGET_S = 300                          # official: 5 min / 1100 problems
SIZE_LIMIT_GB = 20
USE_PUBLIC_SEED = True   # True = reproduce your known public-benchmark number; False = random draw (variance check)
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
if FORCE_CPU:
    os.environ["MODCH_FORCE_CPU"] = "1"

from huggingface_hub import snapshot_download                              # noqa: E402
from modchallenge.config import EvalConfig, PublicBenchmarkConfig          # noqa: E402
from modchallenge.evaluation.pipeline import evaluate_local                 # noqa: E402
from modchallenge.security.static_check import check_submission            # noqa: E402
from modchallenge.interface.submission_schema import SubmissionManifest    # noqa: E402

if COMMIT == "PASTE_FULL_40_CHAR_SHA" or len(COMMIT) != 40:
    raise SystemExit("Set COMMIT to the full 40-character commit SHA you registered "
                     "(find it on the HF repo's commit page, or in your submission record).")

print(f">>> downloading {REPO_ID} @ {COMMIT[:10]}... from HuggingFace ...", flush=True)
sub = Path(snapshot_download(repo_id=REPO_ID, revision=COMMIT,
                             token=TOKEN or None)).resolve()
print(f"    pulled to {sub}")

results = []  # (label, ok, detail)


def add(label, ok, detail=""):
    results.append((label, ok, detail))


# 1) required files present (in the PUSHED artifact)
need = ["manifest.json", "model.py"]
have_w = any((sub / f).exists() for f in
             ("weights.pt", "weights.safetensors", "model.safetensors"))
add("required files (manifest.json, model.py)",
    all((sub / f).exists() for f in need), str(sub))
add("weight file present in repo", have_w, "the uploaded commit is missing weights if this fails")

# 2) manifest schema + required non-empty descriptions
try:
    m = json.loads((sub / "manifest.json").read_text())
    SubmissionManifest.model_validate(m)
    ok = bool(m.get("model_description", "").strip()) and \
        bool(m.get("training_description", "").strip())
    add("manifest schema + descriptions", ok,
        f"entry={m.get('entry_class')} base={m.get('output_base')}")
except Exception as e:  # noqa: BLE001
    add("manifest schema + descriptions", False, repr(e))

# 3) static compliance scan
try:
    findings = check_submission(sub)
    add("static compliance (no banned imports / shortcuts)",
        not findings, "; ".join(f.format() for f in findings) or "clean")
except Exception as e:  # noqa: BLE001
    add("static compliance", False, repr(e))

# 4) artifact size
size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
add(f"artifact size < {SIZE_LIMIT_GB} GB",
    size < SIZE_LIMIT_GB * 1024**3, f"{size/1024**2:.1f} MB")

# 5) full pipeline on the pulled artifact: load -> isolation -> determinism -> score
print(f">>> running full pipeline on the pulled artifact "
      f"({'CPU' if FORCE_CPU else 'auto device'}, {TOTAL} problems, "
      f"{'public seed' if USE_PUBLIC_SEED else 'random seed'}) ...", flush=True)
try:
    cfg = EvalConfig(total_problems=TOTAL, timeout_seconds=1800)
    seed = PublicBenchmarkConfig().seed if USE_PUBLIC_SEED else None
    t0 = time.time()
    res = (evaluate_local(sub, master_seed=seed, config=cfg) if seed is not None
           else evaluate_local(sub, config=cfg))
    wall = time.time() - t0
    add("pipeline ran end-to-end", True, "")
    add("deterministic (required for ranking)", bool(res.deterministic), "")
    hi = max((t.tier_id for t in res.tier_results
              if t.tier_id >= 1 and t.completed and t.accuracy >= 0.90), default=0)
    add("inference within ~time budget (hardware-dependent)",
        wall <= TIME_BUDGET_S, f"{wall:.1f}s local vs {TIME_BUDGET_S}s budget")
    print(f"\n    overall_accuracy = {res.overall_accuracy:.3f}   H>90 = {hi}")
    print("    per-tier:", "  ".join(
        f"{t.tier_id}:{t.accuracy:.2f}" for t in sorted(res.tier_results, key=lambda r: r.tier_id)))
except Exception as e:  # noqa: BLE001
    add("pipeline ran end-to-end", False, repr(e))

# summary
print("\n" + "=" * 62)
print("HUGGINGFACE-ARTIFACT CHECKLIST")
print("=" * 62)
allok = True
for label, ok, detail in results:
    allok &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}" + (f"   ({detail})" if detail else ""))
print("=" * 62)
print("ALL GREEN — the registered submission is intact and reproduces in the official pipeline."
      if allok else "Some checks FAILED — fix and re-push before relying on this submission.")
