# Learned Modular Multiplication

**A small autoregressive "scratchpad" transformer that learns `(a · b) mod p` as an operation — and clears the third tier of the SAIR Modular Arithmetic Challenge.**

Marcos Costa Santos Carreira · Team MCSC (MAC01-T00008)

---

## Result

On the challenge's official public pipeline (100 problems/tier), the submitted model scores:

| tier 1 | tier 2 | tier 3 | H>90 | overall accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 1.00 | 0.99 | **0.93** | **3** | **0.306** |

Deterministic, ~56 s for 1100 problems on CPU. Submission: HuggingFace [`MCarreira/recurrent-v3-ship`](https://huggingface.co/MCarreira/recurrent-v3-ship).

The intellectual core is a **generalisation** result: on tier-3 primes held out of training entirely, the model reaches exact-match **0.82 ± 0.03 across three seeds**, with held-out ≈ train — tier 3 draws from thousands of primes, so this cannot be a lookup.

## The idea in one paragraph

A prime-conditioned feed-forward network memorises per-prime multiplication tables and plateaus (~0.5 on tier 2, floor on tier 3), unmoved by capacity, custom data, or reciprocal features. This isn't a tuning problem: Peng, Narayanan & Papadimitriou (2024) prove a bounded-depth transformer cannot compose functions over large domains, and name multi-digit multiplication as exactly such a composition. The fix is to supply depth through **recurrence** — train a small causal decoder to *write out the arithmetic* as a chain-of-thought, Horner-style, reducing as it goes and never forming the full product:

```
acc ← 0
for each digit b_i of b, most-significant first:
    acc ← (acc·10 + a·b_i) mod p
R ← acc        # the last acc is the answer
```

Each generated token is another effective layer, so the decode loop computes what one forward pass provably cannot. The steps are training targets, not hand-coded logic: randomise the weights and the model emits noise. A richer variant (an explicit reduce step, Abacus digit-position embeddings, and a training mix reweighted toward tier 3) then pushes the no-holdout model over the 90% bar.

Full argument, ablations, and the compliance case: **[WRITEUP.md](WRITEUP.md)** (or [docs/WRITEUP.pdf](docs/WRITEUP.pdf)). One-page summary: **[ONE_PAGER.md](ONE_PAGER.md)**.

## Repository layout

```
WRITEUP.md / docs/WRITEUP.pdf   the methods note (full write-up)
ONE_PAGER.md                    one-page summary
EXPERIMENTS.md                  experiment log: every approach tried, what worked
dev/train_recurrent.py          the trainer (scratchpad, curriculum, anneal, band weights)
dev/data.py                     prime pools + balanced/curriculum batch sampling
models/recurrent_v2/            interleaved-Horner architecture (the generalisation model)
models/recurrent_v3_ship/       + reduce-step + Abacus embeddings (the H>90=3 submission)
run_recurrent_*.py              argument-free runners (open in an IDE, press Run)
run_check_submission*.py        official-pipeline verification (local + HuggingFace)
run_upload_hf.py                push a submission dir to HuggingFace
```

## Reproducing

**Training** is self-contained (PyTorch + NumPy). On Apple Silicon / CUDA / CPU:

```bash
pip install torch numpy
python run_recurrent_v3_ship.py     # or run_recurrent_v3_ship_auto.py for chunked, memory-safe runs
```

The runners are argument-free — edit the `CONFIG` block at the top and run. Training writes a best checkpoint, a resume snapshot, and a `*.metrics.csv` you can watch for plateaus.

**Official scoring / submission** (`run_check_submission*.py`, `run_upload_hf.py`) additionally requires the challenge's `modchallenge` harness, distributed with the [SAIR Modular Arithmetic Challenge](https://competition.sair.foundation/) — install it so `modchallenge` is importable, then point `SUBMISSION` / `REPO_ID` at the model dir.

## Compliance

The answer is produced by trained parameters (weight perturbation collapses it). Only per-argument `a%p`, `b%p` reductions; the scratchpad steps are training targets, not forward-pass logic; no symbolic-math libraries, no hand-coded long multiplication/division, no Barrett/Montgomery/CRT, no input-indexed lookups; deterministic on CPU.

## License

Code under the [MIT License](LICENSE). The write-up (`WRITEUP.md`, `ONE_PAGER.md`, `docs/`) © 2026 Marcos Costa Santos Carreira.

*Experiments, code, and a first draft of the write-up were produced with the assistance of a large language model (Claude), under the author's direction and verification.*
