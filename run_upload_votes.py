"""Push the two self-consistency VOTING variants to HuggingFace (slots 2 & 3).  PyCharm run.

Slot 1 (MCarreira/recurrent-raw, greedy, H>90=2) stays registered and UNTOUCHED. These two are
the same trained weights with inference-time majority voting baked into their manifests
(votes=7; temp 0.5 and 0.8; seed 1234 for determinism). Best-per-team scoring => pure upside:
if voting turns tier-3's stochastic errors around, one of these clears 0.90 -> H>90=3; if not,
slot 1 still holds H>90=2. Uploads manifest.json + model.py + weights.pt only.

After it prints each repo_id + commit_hash:
  1) Playground-test each (repo + commit + your HF token, full 1100) — read tier3.
  2) Register them as your 2nd and 3rd models on the My Submission page.
"""
import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"  # classic LFS path; xet hung on new-repo pushes before

from huggingface_hub import HfApi

# submission_dir -> repo_id
UPLOADS = {
    "models/recurrent_raw_vote05": "MCarreira/recurrent-raw-vote05",
    "models/recurrent_raw_vote08": "MCarreira/recurrent-raw-vote08",
}
PRIVATE = False   # PUBLIC — the eval reads the repo; best-per-team means no downside to shipping
TOKEN   = None    # None -> use your cached `huggingface-cli login`

REPO = Path(__file__).resolve().parent
api = HfApi(token=TOKEN)
results = []
for sub, repo_id in UPLOADS.items():
    folder = (REPO / sub).resolve()
    print(f"\n[upload] {sub} -> {repo_id} ...", flush=True)
    api.create_repo(repo_id, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["manifest.json", "model.py", "weights.pt"],
    )
    sha = api.repo_info(repo_id).sha
    results.append((repo_id, sha))

print("\n" + "=" * 60)
for repo_id, sha in results:
    print(f"  repo_id     : {repo_id}")
    print(f"  commit_hash : {sha}")
    print("-" * 60)
print("Playground-test each (full 1100), read tier3, then register as models 2 & 3.")
print("If a vote slot clears tier3 >= 0.90 -> H>90=3; best-per-team keeps it. Slot 1 is your floor.")
