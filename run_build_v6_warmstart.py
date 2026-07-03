"""Build the recurrent_v6 (20M, 14-layer) warm-start from recurrent_v5 (11.6M, 8-layer).  PyCharm run.

v6 is v5 made DEEPER (8 -> 14 layers, same 384 width) to buy per-step fidelity on the
20-digit multiply-reduce -- the bottleneck that pinned v5's tier 5 at ~0.80 (per-step
~0.989 over 20 steps; clearing 0.90 needs ~0.995/step). We do NOT train the deeper net
cold: we transplant v5 and add refinement depth on top.

  * embeddings (token / position / Abacus place), the final LayerNorm and the output
    head are identical in shape -> copied verbatim;
  * v5's 8 trained encoder layers -> v6 layers 0-7, verbatim (the learned circuit);
  * v6's 6 NEW layers 8-13 are initialised NEAR-IDENTITY: their residual-branch output
    projections (self_attn.out_proj and the FFN's linear2) are zeroed, so each new layer
    outputs x + 0. Because the incoming x is already LayerNorm-normalised and the norms'
    affine params start at (weight 1, bias 0), norm(x+0) ~= x -- so at init the 14-layer
    net reproduces v5's function almost exactly, then the top layers learn to sharpen it.

Writes models/recurrent_v6/weights.pt at step 0. The auto driver runs this if the seed
is missing. Requires a trained models/recurrent_v5/weights.pt.
"""
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "dev"))

V5_CKPT = REPO / "models" / "recurrent_v5" / "weights.pt"
V6_DIR = REPO / "models" / "recurrent_v6"
OUT = V6_DIR / "weights.pt"

NEW_LAYERS_ON_TOP = 6          # 8 (v5) + 6 = 14
# residual-branch output params to zero for a near-identity new layer:
IDENTITY_ZERO = ("self_attn.out_proj.weight", "self_attn.out_proj.bias",
                 "linear2.weight", "linear2.bias")


def load_arch(d):
    spec = importlib.util.spec_from_file_location("rec_model_v6", Path(d) / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    if not V5_CKPT.exists():
        raise SystemExit(f"need trained v5 weights at {V5_CKPT} — train v5 first.")
    v6 = load_arch(V6_DIR)
    ck = torch.load(V5_CKPT, map_location="cpu", weights_only=True)
    v5_sd, v5_cfg = ck["state_dict"], ck["config"]
    n_old = int(v5_cfg["num_layers"])
    n_new = n_old + NEW_LAYERS_ON_TOP
    print(f"v5 config: {v5_cfg}")
    print(f"v6: {n_new} layers (= {n_old} copied + {NEW_LAYERS_ON_TOP} near-identity), "
          f"d_model={v5_cfg['d_model']}, seq_len={v6.SEQ_LEN}")

    model = v6.ScratchpadDecoder(d_model=v5_cfg["d_model"], nhead=v5_cfg["nhead"],
                                 num_layers=n_new, dim_ff=v5_cfg["dim_ff"],
                                 seq_len=v6.SEQ_LEN)
    new_sd = model.state_dict()
    copied, ident, kept = 0, 0, 0
    for k, v in new_sd.items():
        if k.startswith("encoder.layers."):
            idx = int(k.split(".")[2])
            sub = k.split(".", 3)[3]                       # e.g. "self_attn.out_proj.weight"
            if idx < n_old:                                # transplant v5 layer
                new_sd[k] = v5_sd[k].clone(); copied += 1
            elif sub in IDENTITY_ZERO:                     # new layer: zero residual output
                new_sd[k] = torch.zeros_like(v); ident += 1
            else:
                kept += 1                                  # new layer: keep default init
        elif k in v5_sd and v5_sd[k].shape == v.shape:     # embeddings / ln / head
            new_sd[k] = v5_sd[k].clone(); copied += 1
        else:
            kept += 1

    model.load_state_dict(new_sd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"copied {copied} tensors from v5; zeroed {ident} residual-output tensors "
          f"({NEW_LAYERS_ON_TOP} near-identity layers); kept {kept} at default init.")
    print(f"v6 params: {n_params/1e6:.2f}M")

    V6_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "bands": ["tier1", "tier2", "tier3", "tier4", "tier5"],
                "h90": -1, "frontier_val": -1.0, "step": 0}, OUT)
    for suf in (".latest", ".latest.opt", ".opt", ".metrics.csv"):
        sp = Path(str(OUT) + suf)
        if sp.exists():
            sp.unlink()
    print(f"\nwrote v6 warm-start seed -> {OUT} (step 0). Next: run run_recurrent_v6_auto.py")


if __name__ == "__main__":
    main()
