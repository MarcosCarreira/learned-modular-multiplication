# Show your work: a scratchpad transformer that reduces raw operands mod p in-network — compliant H>90=2, and a systematic-drift wall at tier 3

**Marcos Costa Santos Carreira** · Team MCSC (MAC01-T00008)

**Summary (compliant).** A small autoregressive decoder takes **raw `(a, b, p)`** and *writes out the arithmetic* — reducing the full-width operands mod p **inside the network** (the organisers ruled that reduction is the intended core of the task) — learning `a·b mod p` as an operation rather than memorising per-prime tables. It factors into one learned primitive `cell(acc, m, digit) = (acc·R + m·digit) mod p` on a fixed two-phase schedule: reduce `a mod p` over `a`'s raw digits, then multiply over `b`'s raw digits. On the official pipeline: tier 1 **1.00**, tier 2 **0.97**, tier 3 **0.69** → **`H>90 = 2`, OA 0.28**, deterministic, 11 s / 1100. Submission: HuggingFace `MCarreira/recurrent-raw` @ `f9a757ab…`.

**Where the wall is: systematic drift, not capacity.** Tier 3 doesn't clear 90%, and the diagnosis is the contribution. The ceiling is rollout drift, confirmed three independent ways — all negative: a 2× deeper model *lowered* the frontier and overfit the eval; harder on-policy training (scheduled sampling 0.30→0.55) plateaued; and inference-time self-consistency (majority vote over 7 sampled rollouts, submitted as two of the three allowed models) left tier 3 flat at **0.69 / 0.68 / 0.69** across greedy and temperatures 0.5/0.8. Raising the temperature diversified the rollouts and the vote still didn't move — so the tier-3 errors are *systematic* (the same cases reduced wrong every sample), not stochastic noise that voting could average out. Independently corroborated by the concurrent neural-horner work.

> **The sections below** report four- and five-tier "scaling" results on a **reduced-operand sub-problem** (`a%p, b%p` pre-reduced) that was later ruled **out of spec** — kept as learnability science, clearly flagged, not as valid competition entries.

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
| + widened to 20 digits (v5) | — | — | — | t2 0.98 / t3 0.90 / t4 0.88 / **t5 0.80** → H>90=3, OA 0.47 |
| + doubled capacity, 2× depth (v6) | — | — | — | t3 0.90 / t4 0.86 / **t5 0.72** → H>90=3 — *capacity made it worse* |

**The scaling, one tier further.** Widening the *same* recipe once more (WIDTH 10→20, twenty Horner steps) reaches tier 5 (p < 2⁶⁴, ≤20-digit operands). Official benchmark: tier 2 0.98 / tier 3 0.90 / tier 4 0.88 / **tier 5 0.80** — the operation still generalising to unseen primes a full doubling of operand length further, the curve 0.98/0.90/0.88/0.80 across 2/5/10/20 digits declining only ~0.18 over a 16× range rather than collapsing. Honest cost: tier 5's 0.80 is below the 90% bar (so this model is H>90=3 and the 10-digit model stays the submission), and its tier 3/4 sit a little below the 10-digit model's — a fixed ~11M-parameter budget spread across five tiers of longer sequences. The claim is "the same recipe *reaches* 20 digits with graceful degradation," not "wider is uniformly better."

**Capacity is not the lever (a clean negative).** Asked directly whether more parameters clear tier 5, we doubled the model (2× depth, ~20M) — and it scored *worse*: official tier 5 0.72 vs the smaller model's 0.80, with the dev→official gap widening from ~0.06 to ~0.14. More capacity overfit the eval rather than generalising, so the 20-digit ceiling is **drift/generalisation-limited, not capacity-limited** (corroborating the concurrent neural-horner work). The remaining lever is exposure-bias / on-policy training, not scale.

**Compliance.** The compliant model takes **raw `(a, b, p)`** and reduces the full-width operands **in-network** — no `a%p`/`b%p` in preprocessing (the ruling: pre-reduction offloads the core of the task; the `digit_transformer` reference example suggesting otherwise was confirmed an error, not a precedent). Answer from trained parameters (weight perturbation collapses it); scratchpad steps are training targets, not forward-pass logic; no symbolic libraries, no hand-coded long multiplication/division, no Barrett/Montgomery/CRT, no input-indexed lookups; deterministic on CPU (the self-consistency variants seed their sampling). The reduced-operand `v2/v4/v5` line is out of spec, kept only as learnability science.

**One honest note.** The compliant standing is `H>90 = 2`: tiers 1–2 clear, tier 3 lands 0.69 official. The tier-3 wall is diagnosed (systematic drift, three converging negatives), not merely observed — breaking it needs a lever that changes what the parameters compute on the drift-prone cases (expert-relabel DAgger, a shorter reduction rollout), not more scale, more on-policy, or a vote.

**Links.** Compliant submission: `MCarreira/recurrent-raw` (H>90=2) @ `f9a757ab…`; self-consistency variants: `recurrent-raw-vote05` / `-vote08`. Sub-problem models (out of spec): `recurrent-v4` (H>90=4), `recurrent-v5` (tier 5 0.80). Full methods note and argument-free reproduction runners (`run_*.py`, `modchallenge evaluate-hf`) in the team repo.

---

*Experiments, code, and a first draft of this note were produced with the assistance of a large language model (Claude), under the author's direction and verification.*
