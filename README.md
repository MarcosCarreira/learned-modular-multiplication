# Learned Modular Multiplication

**A small autoregressive "scratchpad" transformer that learns `(a · b) mod p` as an operation — and clears the first four tiers of the SAIR Modular Arithmetic Challenge, with generalisation that scales nearly flat across operand length.**

Marcos Costa Santos Carreira · Team MCSC (MAC01-T00008)

---

## Result

On the challenge's official pipeline (100 problems/tier), the submitted model scores:

| tier 1 | tier 2 | tier 3 | tier 4 | H>90 | overall accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.00 | 0.97 | 0.95 | **0.96** | **4** | **0.40** |

Deterministic, **5.1 s for 1100 problems** on the official evaluation hardware (≈60× under the 300 s budget). Submission: HuggingFace [`MCarreira/recurrent-v4`](https://huggingface.co/MCarreira/recurrent-v4).

The intellectual core is a **generalisation-and-scaling** result. On primes held out of training entirely, the model generalises (it cannot be a lookup: each tier draws from thousands-to-billions of primes) — tier-3 exact-match **0.82 ± 0.03 across three seeds**, held-out ≈ train. And the same recipe, merely widened (one Horner step per operand digit), carries that generalisation across a **doubling of operand length nearly flat**: held-out tier 3 0.93 → tier 4 0.905, within ~0.02. A memorised table falls off a cliff the instant operands grow; a learned algorithm barely bends. That near-flat scaling — not the tier index — is the result.

**Pushed one tier further, to twenty digits.** Widening the *same* recipe once more (`WIDTH 10 → 20`, twenty Horner steps) reaches tier 5 (p < 2⁶⁴, ≤20-digit operands). On the official benchmark this 20-digit model scores **tier 2 0.98 / tier 3 0.90 / tier 4 0.88 / tier 5 0.80** — the operation still generalising to unseen primes a full doubling of operand length further, declining only ~0.18 across a 16× range of operand magnitude rather than collapsing. Two honest notes: tier 5's 0.80 is below the 90% bar (so this model is `H>90 = 3` and the 10-digit model remains the submission), and its tier 3/4 sit a little under the 10-digit model's because a fixed ~11M-parameter budget spreads thin across five tiers of longer sequences. The claim is *"the same recipe reaches 20 digits with graceful degradation,"* not *"wider is uniformly better"* — but the gentle, unbroken slope is the scaling claim in its strongest form. Model: [`MCarreira/recurrent-v5`](https://huggingface.co/MCarreira/recurrent-v5).

**Is the 20-digit ceiling capacity or drift? A depth test says drift.** We ran the direct experiment — the *same* recipe made deeper (8→14 layers, ~11M→~20M, warm-started, tier-5-focused, annealed) to ask whether more parameters clear tier 5. They do not: on the official benchmark the 2× model scored tier 5 **0.72 — *below* the smaller model's 0.80** — and the gap between its in-distribution dev score and the secret-seed official score *widened* with size (~0.06 → ~0.14). The extra capacity went into overfitting the eval, not generalising further. So the twenty-digit ceiling is **drift/generalisation-limited, not capacity-limited** — corroborating the concurrent [neural-horner](https://github.com/Robby955/neural-horner) work — and the remaining lever is exposure-bias / on-policy training (scheduled sampling, DAgger), not scale. A clean negative, and the sharpest single result about *where* the wall is. Model: `MCarreira/recurrent-v6`.

## The idea in one paragraph

A prime-conditioned feed-forward network memorises per-prime multiplication tables and plateaus (~0.5 on tier 2, floor on tier 3), unmoved by capacity, custom data, or reciprocal features. This isn't a tuning problem: Peng, Narayanan & Papadimitriou (2024) prove a bounded-depth transformer cannot compose functions over large domains, and name multi-digit multiplication as exactly such a composition. The fix is to supply depth through **recurrence** — train a small causal decoder to *write out the arithmetic* as a chain-of-thought, Horner-style, reducing as it goes and never forming the full product:

```
acc ← 0
for each digit b_i of b, most-significant first:
    acc ← (acc·10 + a·b_i) mod p
R ← acc        # the last acc is the answer
```

Each generated token is another effective layer, so the decode loop computes what one forward pass provably cannot. The steps are training targets, not hand-coded logic: randomise the weights and the model emits noise. A richer variant (an explicit reduce step, Abacus digit-position embeddings) clears tier 3, and **widening the very same recipe** — WIDTH 5→10, one Horner step per digit — clears tier 4 too, at linear cost in operand length (exactly the composition depth the depth-limit theorem demands). Because the cost is linear and the per-step operation never changes, the method scales by construction; the open question above tier 4 is accuracy, not the time budget — and pushing to twenty digits answers it with tier 5 0.80, a gentle decline rather than a cliff.

Full argument, ablations, and the compliance case: **[WRITEUP.md](WRITEUP.md)** (or [docs/WRITEUP.pdf](docs/WRITEUP.pdf)). One-page summary: **[ONE_PAGER.md](ONE_PAGER.md)**.

## Repository layout

```
WRITEUP.md / docs/WRITEUP.pdf   the methods note (full write-up)
ONE_PAGER.md                    one-page summary
EXPERIMENTS.md                  experiment log: every approach tried, what worked
dev/train_recurrent.py          the trainer (scratchpad, curriculum, anneal, band weights)
dev/data.py                     prime pools + balanced/curriculum batch sampling
models/recurrent_v2/            interleaved-Horner architecture (the generalisation model)
models/recurrent_v3_ship/       + reduce-step + Abacus embeddings (the H>90=3 model)
models/recurrent_v4/            same recipe widened to 10-digit operands (the H>90=4 submission)
models/recurrent_v5/            widened again to 20-digit operands (the scaling result: tier5 0.80)
run_recurrent_*.py              argument-free runners (open in an IDE, press Run)
run_check_submission*.py        official-pipeline verification (local + HuggingFace)
run_upload_hf.py                push a submission dir to HuggingFace
```

## Reproducing

**Training** is self-contained (PyTorch + NumPy). On Apple Silicon / CUDA / CPU:

```bash
pip install torch numpy
python run_recurrent_v4.py          # the H>90=4 model; _auto.py variants do chunked, memory-safe runs
```

The runners are argument-free — edit the `CONFIG` block at the top and run. Training writes a best checkpoint, a resume snapshot, and a `*.metrics.csv` you can watch for plateaus.

**Official scoring / submission** (`run_check_submission*.py`, `run_upload_hf.py`) additionally requires the challenge's `modchallenge` harness, distributed with the [SAIR Modular Arithmetic Challenge](https://competition.sair.foundation/) — install it so `modchallenge` is importable, then point `SUBMISSION` / `REPO_ID` at the model dir.

## Compliance

The answer is produced by trained parameters (weight perturbation collapses it). Only per-argument `a%p`, `b%p` reductions; the scratchpad steps are training targets, not forward-pass logic; no symbolic-math libraries, no hand-coded long multiplication/division, no Barrett/Montgomery/CRT, no input-indexed lookups; deterministic on CPU.

## License

Code under the [MIT License](LICENSE). The write-up (`WRITEUP.md`, `ONE_PAGER.md`, `docs/`) © 2026 Marcos Costa Santos Carreira.

*Experiments, code, and a first draft of the write-up were produced with the assistance of a large language model (Claude), under the author's direction and verification.*
