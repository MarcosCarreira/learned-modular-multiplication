"""Recurrent scratchpad — BASE-2 (bit-serial) RADIX ABLATION.

Identical architecture and algorithm to the base-10 scratchpad (recurrent_v3/v5), with
one axis changed: the Horner loop runs in BASE 2, one BIT per step, instead of base 10 one
DIGIT per step. This is the ablation the neural-Horner comparison (Sneiderman, base-2) and
our own base-1000 negative both point to: does spending MORE, smaller steps (base-2: WIDTH
bits) hurt via rollout drift vs FEWER, still-learnable steps (base-10: WIDTH/3.3 digits)?

WIDTH=16 bits covers tiers 1-3 (p < 2^16). At tier 3 this is ~16 Horner steps vs base-10's
~5 — so if the drift theory is right, base-2 tier-3 should trail base-10 tier-3 (~0.9+).
Per step:  m_i = a * bit_i ;  t_i = acc_{i-1}*2 + m_i ;  acc_i = t_i mod p.  Final acc = a*b mod p.
Compliance unchanged: worked steps are TRAINING targets, answer is the model's generated acc,
a%p/b%p legal reductions, no hand-coded arithmetic. Output is the answer as decimal digits.
NOT a submission — a controlled ablation to quantify steps-vs-drift in one harness.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 16                         # BITS of a%p, b%p, p, acc  (p < 2^16 covers tiers 1-3)
RADIX_DIGITS = 1                   # 1 bit per Horner step
RADIX = 2 ** RADIX_DIGITS          # 2  <-- the ablation: base 2, not base 10
NB = -(-WIDTH // RADIX_DIGITS)     # 16 Horner steps (one per bit of b)
MWIDTH = WIDTH + RADIX_DIGITS      # bits of m_i = a * bit        (17)
TWIDTH = WIDTH + RADIX_DIGITS + 1  # bits of t_i = acc*2 + m      (18)
LIMIT = 2 ** WIDTH  # 2^16 — primes at or above this are out of range (-> [0])

PAD, BOS, SEP, EOS = 2, 3, 4, 5    # bit tokens are 0,1; markers start at 2
VOCAB = 6
MAXPLACE = max(WIDTH, MWIDTH, TWIDTH)
NO_PLACE = MAXPLACE
NPLACE = MAXPLACE + 1

PROMPT_LEN = 1 + (WIDTH + 1) * 3
GROUP = (MWIDTH + 1) + (TWIDTH + 1) + (WIDTH + 1)
GEN_LEN = GROUP * NB
SEQ_LEN = PROMPT_LEN + GEN_LEN
R_OFF = GROUP * (NB - 1) + (MWIDTH + 1) + (TWIDTH + 1)


def _place_ids() -> list[int]:
    ids = [NO_PLACE]
    for _ in range(3):
        ids += list(range(WIDTH)) + [NO_PLACE]
    for _ in range(NB):
        ids += list(range(MWIDTH)) + [NO_PLACE]
        ids += list(range(TWIDTH)) + [NO_PLACE]
        ids += list(range(WIDTH)) + [NO_PLACE]
    return ids


def to_field(n: int, width: int) -> list[int]:
    out = []
    for _ in range(width):
        out.append(n % 2); n //= 2          # base-2 (LSB-first)
    return out


def from_field(ids) -> int:
    v = 0
    for d in reversed(list(ids)):
        d = int(d)
        v = v * 2 + (d if 0 <= d <= 1 else 0)   # base-2 reconstruct
    return v


def build_prompt(a: int, b: int, p: int) -> list[int]:
    return ([BOS] + to_field(a, WIDTH) + [SEP] + to_field(b, WIDTH)
            + [SEP] + to_field(p, WIDTH) + [SEP])


def build_full(a: int, b: int, p: int) -> list[int]:
    seq = build_prompt(a, b, p)
    acc = 0
    for i in range(NB - 1, -1, -1):
        bit = (b // (RADIX ** i)) % RADIX          # i-th bit of b, MSB first
        m = a * bit
        t = acc * RADIX + m
        acc = t % p
        seq += (to_field(m, MWIDTH) + [SEP] + to_field(t, TWIDTH) + [SEP]
                + to_field(acc, WIDTH) + [SEP])
    seq[-1] = EOS
    return seq


class ScratchpadDecoder(nn.Module):
    def __init__(self, d_model=384, nhead=8, num_layers=8, dim_ff=1024, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.place_emb = nn.Embedding(NPLACE, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)
        self.register_buffer("pos_ids", torch.arange(seq_len), persistent=False)
        self.register_buffer("place_ids", torch.tensor(_place_ids()[:seq_len]), persistent=False)
        self.config = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_ff=dim_ff, seq_len=seq_len)

    def _embed(self, tokens, start=0):
        L = tokens.shape[1]
        pos = self.pos_ids[start:start + L].unsqueeze(0)
        plc = self.place_ids[start:start + L].unsqueeze(0)
        return self.tok_emb(tokens) + self.pos_emb(pos) + self.place_emb(plc)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        x = self._embed(tokens)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
        return self.head(self.ln(self.encoder(x, mask=mask)))

    @torch.no_grad()
    def _generate_uncached(self, prompt, steps):
        mps = (prompt.device.type == "mps" and hasattr(torch, "mps"))
        seq = prompt
        for s in range(steps):
            seq = torch.cat([seq, self.forward(seq)[:, -1, :].argmax(-1, keepdim=True)], dim=1)
            if mps and (s & 31) == 31:
                torch.mps.empty_cache()
        return seq[:, prompt.shape[1]:]

    def _layer_step(self, layer, x, Kc, Vc):
        attn = layer.self_attn
        H, d = attn.num_heads, x.shape[-1]; dh = d // H
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.split(d, dim=-1)
        sh = lambda t: t.view(x.shape[0], -1, H, dh).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if Kc is not None:
            k = torch.cat([Kc, k], dim=2); v = torch.cat([Vc, v], dim=2)
        att = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        Tnew, Tall = q.shape[2], k.shape[2]
        i = torch.arange(Tnew, device=x.device).unsqueeze(1) + (Tall - Tnew)
        j = torch.arange(Tall, device=x.device).unsqueeze(0)
        att = att.masked_fill(j > i, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(x.shape[0], Tnew, d)
        o = F.linear(o, attn.out_proj.weight, attn.out_proj.bias)
        x = layer.norm1(x + o)
        x = layer.norm2(x + layer.linear2(F.gelu(layer.linear1(x))))
        return x, k, v

    @torch.no_grad()
    def generate_cached(self, prompt, steps):
        layers = list(self.encoder.layers)
        Kc = [None] * len(layers); Vc = [None] * len(layers)
        Lp = prompt.shape[1]
        x = self._embed(prompt, start=0)
        for li, lyr in enumerate(layers):
            x, Kc[li], Vc[li] = self._layer_step(lyr, x, None, None)
        nxt = self.head(self.ln(x[:, -1:, :])).argmax(-1)
        gen = [nxt]; pos = Lp
        mps = (prompt.device.type == "mps" and hasattr(torch, "mps"))
        for s in range(steps - 1):
            x = self._embed(nxt, start=pos)
            for li, lyr in enumerate(layers):
                x, Kc[li], Vc[li] = self._layer_step(lyr, x, Kc[li], Vc[li])
            nxt = self.head(self.ln(x[:, -1:, :])).argmax(-1)
            gen.append(nxt); pos += 1
            if mps and (s & 31) == 31:
                torch.mps.empty_cache()
        return torch.cat(gen, dim=1)

    @torch.no_grad()
    def _verify_cache(self, dev):
        try:
            L = max(4, self.seq_len // 4)
            p = torch.randint(0, VOCAB, (3, L), device=dev)
            ok = torch.equal(self._generate_uncached(p, 10), self.generate_cached(p, 10))
            print(f"[KV-cache] {'verified == uncached; fast path on' if ok else 'MISMATCH -> using uncached'}", flush=True)
            return bool(ok)
        except Exception as e:  # noqa: BLE001
            print(f"[KV-cache] error ({e}); using uncached", flush=True); return False

    @torch.no_grad()
    def generate(self, prompt, steps):
        if getattr(self, "_cache_ok", None) is None:
            self._cache_ok = self._verify_cache(prompt.device)
        return (self.generate_cached if self._cache_ok else self._generate_uncached)(prompt, steps)


from modchallenge.interface.base_model import ModularMultiplicationModel  # noqa: E402


class ScratchpadModel(ModularMultiplicationModel):
    def __init__(self):
        self.model = None; self.device = None

    def load(self, model_dir: str) -> None:
        if os.environ.get("MODCH_FORCE_CPU") == "1":
            self.device = torch.device("cpu")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        ckpt = torch.load(Path(model_dir) / "weights.pt", map_location=self.device, weights_only=True)
        self.model = ScratchpadDecoder(**ckpt.get("config", {}))
        self.model.load_state_dict(ckpt["state_dict"]); self.model.to(self.device); self.model.eval()

    def preprocess_a(self, a): return a
    def preprocess_b(self, b): return b
    def preprocess_p(self, p): return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    @torch.no_grad()
    def predict_digits_batch(self, inputs):
        out = [None] * len(inputs); prompts, idx = [], []
        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            if not (2 <= p < LIMIT):
                out[i] = [0]; continue
            prompts.append(build_prompt(int(a_enc) % p, int(b_enc) % p, p)); idx.append(i)
        if idx:
            pt = torch.tensor(prompts, dtype=torch.long, device=self.device)
            gen = self.model.generate(pt, GEN_LEN).cpu().tolist()
            for j, i in enumerate(idx):
                out[i] = [int(d) for d in str(from_field(gen[j][R_OFF:R_OFF + WIDTH]))]
        return [o if o is not None else [0] for o in out]

    def max_batch_size(self): return 256
