# Show your work: chain-of-thought scratchpads clear tier 4 of the Modular Multiplication Challenge — and the generalisation scales

**Marcos Costa Santos Carreira** · Team MCSC (MAC01-T00008)

**Summary.** A small autoregressive decoder that *writes out the arithmetic* — rather than mapping `(a, b, p)` straight to an answer — learns `a·b mod p` as an operation instead of memorising per-prime multiplication tables, and **clears the first four tiers** of the benchmark. On the official pipeline (100 problems/tier) the submitted model scores tier 1 **1.00**, tier 2 **0.97**, tier 3 **0.95**, tier 4 **0.96** → **`H>90 = 4`, overall accuracy 0.40**, deterministic, **5.1 s for 1100 problems** on the official hardware (≈60× under budget). Submission: HuggingFace `MCarreira/recurrent-v4`. The sharper result is the *scaling*: widening the identical recipe one digit at a time carries unseen-prime generalisation across a doubling of operand length nearly flat (held-out tier 3 0.93 → tier 4 0.905) — what a learned operation does and a memorised table cannot.

**The failure everyone hits first.** A prime-conditioned feed-forward encoder plateaus at tier-2 exact-match ≈ 0.5 and floors on tier 3, and stays there under more parameters, custom data distributions, and reciprocal/Barrett features — consistent with Lauter et al. (2024), who report seq2seq and circular-regression both "demonstrate poor generalization." This isn't a tuning problem: Peng, Narayanan & Papadimitriou (2024) prove a bounded-depth Transformer cannot compose functions over large domains, and they name multi-digit multiplication as exactly such a composition. A single forward pass lacks the depth.

**The fix: supply depth through recurrence.** Train a small causal decoder to generate the multiplication as a chain-of-thought, Horner-style, reducing as it goes and **never forming the full product**:

```
acc ← 0
for each digit b_i of b, most-significant first:
    acc ← (acc·10 + a·b_i) mod p
R ← acc        # the last acc is the answer
```

Each generated token is another effective layer, so the decode loop computes what one pass provably cannot — and Peng et al.'s own CoT theorem says the trace must scale with the composition depth, which a per-digit Horner scratchpad does exactly. The worked steps are *training targets*, not hand-coded logic: randomise the weights and the model emits noise; the answer is computed step by step from trained parameters.

**The evidence it learned, not memorised.** On tier-3 primes **held out of training entirely**, the model reaches exact-match **0.82 ± 0.03 across three seeds**, with held-out ≈ train. Tier 3 draws from thousands of primes, so an unseen-prime 0.82 cannot be a lookup. A richer variant — an explicit `t = acc·10 + m` reduce step, Abacus digit-position embeddings (McLeish et al. 2024) — clears tier 3, and then **the same recipe merely widened** (WIDTH 5→10, one Horner step per digit) clears tier 4: on held-out unseen primes the generalisation carries across the operand-length doubling **within ~0.02** (tier 3 0.93 → tier 4 0.905), and on the official hardware the model scores tier 4 0.96, `H>90=4`, OA 0.40 in 5.1 s. Cost is linear in operand length — the composition depth the depth-limit theorem demands — so the method scales by construction.

| model | t2 unseen | t3 unseen | t4 unseen | official |
|---|---|---|---|---|
| feed-forward encoder | 0.13 | ~0.05 | — | floor |
| Horner scratchpad | 0.83 | **0.82 ± 0.03** | — | — |
| + reduce-step + Abacus (v3) | — | — | — | t3 0.93 → H>90=3, OA 0.306 |
| **+ widened to 10 digits (v4)** | 0.96 | 0.93 | **0.905** | **t4 0.96 → H>90=4, OA 0.40** |

**Compliance.** Answer from trained parameters (weight perturbation collapses it); only per-argument `a%p`, `b%p` reductions; scratchpad steps are training targets, not forward-pass logic; no symbolic libraries, no hand-coded long multiplication/division, no Barrett/Montgomery/CRT, no input-indexed lookups; deterministic on CPU.

**One honest margin note.** The tier-4 result is a single converged seed (stable across eight post-anneal evaluations); multi-seed bands, as on tier 3, would harden it against the secret-seed final eval. The robust claim is the *generalisation-and-scaling* — held-out ≈ train on unseen primes, carried nearly flat across operand length — of which the cleared tiers are a consequence.

**Links.** Submission: `MCarreira/recurrent-v4`. Full methods note and argument-free reproduction runners (`run_*.py`, `modchallenge evaluate-hf`) in the team repo.

---

*Experiments, code, and a first draft of this note were produced with the assistance of a large language model (Claude), under the author's direction and verification.*
