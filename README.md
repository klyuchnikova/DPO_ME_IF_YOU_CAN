# DPO Me If You Can

DPO Me If You Can — efficient token-weighted DPO for `Qwen2.5-0.5B-Instruct` under limited compute.

We study whether cheap token-importance approximations can match the *spirit* of TI-DPO (Token-Importance Guided Direct Preference Optimization) without online gradient attribution at every training step. Full TI-DPO code is not public; this repository is our own replication and ablation.

---

## 1. Motivation

Standard DPO optimizes a sequence-level preference objective: every response token contributes equally to

$$
\Delta_{\mathrm{DPO}} = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \quad,\quad L_{\mathrm{DPO}} = -\log \sigma(\beta \Delta_{\mathrm{DPO}})
$$

TI-DPO argues that some tokens matter more for human preference (e.g. safety phrases, key facts) than filler text. It replaces the sum with token weights $w_t$:

$$
\Delta_{\mathrm{weighted}} = \sum_t w^{w}_{t} \left[\log \pi_\theta(y^{w}_{t} \mid \cdot) - \log \pi_{\mathrm{ref}}(y^{w}_{t} \mid \cdot)\right] - \sum_t w^{l}_{t} \left[\log \pi_\theta(y^{l}_{t} \mid \cdot) - \log \pi_{\mathrm{ref}}(y^{l}_{t} \mid \cdot)\right]
$$

In the paper, $w_t$ combines gradient attribution and a Gaussian positional prior:

$$
W = \lambda I_{\mathrm{grad}} + (1-\lambda) P_{\mathrm{gaussian}}
$$

Computing $I_{\text{grad}}$ online (backward through input embeddings every step) is expensive on a single GPU.

Our goal:

Make token-importance DPO cheaper for small LLMs by replacing online gradient attribution with cached or approximate token-importance weights — while keeping LoRA DPO training practical on one V100.


We do not reproduce TI-DPO's triplet loss or full PPO pipeline.

Research questions:

1. Can token-weighted DPO improve over vanilla DPO on small LLMs?
2. Can surprisal ($-\log \pi_{\text{ref}}$) work as a almost-free proxy?
3. Can cached gradients match online attribution at near-DPO train speed?
4. Does the answer depend on the dataset (general helpfulness vs safety-style preferences)?

---

## 2. Literature and related work

### 2.1 Reinforcement from human feedback and DPO

Reinforcement Learning from Human Feedback (RLHF) turned alignment into an optimization problem: collect human comparisons over model outputs, train a reward model rϕ to predict those comparisons, then optimize a generation policy πθ to maximize the learned reward under a KL constraint to a reference policy πref (PPO-based and later variants). A simplifying alternative, Direct Preference Optimization (DPO) [Rafailov et al., 2023], shows that under the Bradley–Terry preference model the reward signal can be expressed as a log-ratio of policy to reference probabilities, and that a closed-form objective equivalent to an implicit reward can be used instead of an explicit reward model + RL loop.

DPO objective (sequence-level):
\[
\Delta_{\text{DPO}}(x,y_w,y_l)=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\]
\[
L_{\text{DPO}}=-\log\sigma(\beta\Delta_{\text{DPO}})
\]
This objective avoids an explicit reward model and is easy to implement using two forward passes (policy and reference) and a simple sigmoid loss. DPO thus became a practical RLHF alternative for fine-tuning LLMs with preference pairs.

### 2.2 Token-level concerns and TI‑DPO

A limitation of DPO and sequence-level RLHF is granularity: they treat each token equally, though human preferences often hinge on a few decisive tokens (safety instructions, factual claims). Token-level variants of DPO and RLHF have been proposed: they decompose sequence reward into per-token contributions and reweight tokens to focus learning [TDPO, TIS-DPO, others]. The TI‑DPO paper (the main paper motivating this project) proposes two key ideas to achieve robust token-level alignment:

1. hybrid token importance: combine gradient-based attribution \(I_{\text{grad}}\) (how much each token embedding influences a target scalar) with a Gaussian positional prior \(P_{\text{gaussian}}\) to correct “lost-in-the-middle” biases; and
2. a triplet-style auxiliary loss to provide structured guidance for intermediate generations.

TI‑DPO weight formula:
\[
W = \lambda I_{\text{grad}} + (1-\lambda)P_{\text{gaussian}}
\]

They show theoretically that weighting non-critical tokens down reduces variance and yields a tighter loss bound versus vanilla DPO; empirically TI‑DPO reports gains on a range of tasks. However, computing gradient attribution online for every sequence can double training cost (extra backward pass per example), which is problematic on limited hardware.

### 2.3 Gradient attribution and positional priors

Gradient attribution methods (e.g., input-gradient norms, integrated gradients) estimate how much each input feature contributes to a scalar target. In text, a common technique is computing \(I_t = \lVert \nabla_{e_t} L_{\text{target}}\rVert_1\) where \(e_t\) is the token embedding and \(L_{\text{target}}\) is a chosen scalar (e.g., final logit or NLL). These are attractive because they directly tie tokens to the model's objective, but they require extra backward work.

Empirical findings also show position biases in LLMs (e.g., “lost-in-the-middle”): attention/importance sometimes peaks at edges and underweights middle tokens (see Liu et al., Lost-in-the-Middle). A Gaussian prior centered on the response can partially correct arbitrary architectural biases and stabilize importance estimation.

### 2.4 Cheap proxies: surprisal and cached importance

Two practical ideas arise:

- Surprisal as proxy: token surprisal under the reference model, i.e. \(-\log\pi_{\text{ref}}(y_t)\), is a cheap content-dependent signal that requires no extra backward passes because ref logprobs are already computed for DPO. Intuitively, surprising tokens may be information-rich and preference-relevant.
- Cached gradient attribution: compute gradient-based token importance once (or infrequently) offline and reuse it during training to avoid repeated backward costs. The tradeoff is staleness: importance computed on a frozen reference may become less accurate as the policy drifts with LoRA updates.

### 2.5 Efficient fine-tuning (LoRA / PEFT) and practical constraints

To run experiments on limited hardware we use parameter-efficient fine-tuning (PEFT / LoRA) to adapt the model by adding small low-rank updates. This keeps memory and compute requirements low and isolates adaptation to a small subset of parameters (q,k,v,o projections typically).

### 2.6 Summary of the gap

TI‑DPO provides a strong conceptual framework for token-level alignment but is computationally heavier due to online attribution and auxiliary triplet losses. Our project focuses on the practical question: can cheap or cached approximations (surprisal, cached gradients, sparse top‑k) recover most of the gains of token-level weighting while staying practical on a single V100?

---

## 3. Methods (detailed)

This chapter defines the loss, the weight variants, normalization, and the exact training/eval pipelines used in experiments.

### 3.1 Notation and base DPO formula

- prompt \(x\)
- preferred (chosen) response \(y_w\)
- less-preferred (rejected) response \(y_l\)
- policy \(\pi_\theta\) (trainable — LoRA adapters enabled)
- reference \(\pi_{\text{ref}}\) (base weights, LoRA disabled)

Per-token log-probabilities for a sequence \(y\) under model \(m\) are \(\log \pi_m(y_t\mid x,y_{<t})\). Define the per-token advantage:
\[
\mathrm{adv}_t(x,y) := \log\pi_\theta(y_t\mid x,y_{<t}) - \log\pi_{\text{ref}}(y_t\mid x,y_{<t})
\]

Weighted-token DPO difference:
\[
\Delta_{\text{weighted}}(x,y_w,y_l) = \sum_{t\in y_w} w^w_t \cdot \mathrm{adv}_t(x,y_w) - \sum_{t\in y_l} w^l_t \cdot \mathrm{adv}_t(x,y_l)
\]
Loss:
\[
L = -\log\sigma(\beta\Delta_{\text{weighted}})
\]
For uniform DPO, \(w_t\equiv 1\).

Normalization: after building raw weights \(\tilde w_t\) for the response tokens we re-normalize so the mean weight across active response tokens equals 1:
\[
w_t = \frac{\tilde w_t}{\frac{1}{T}\sum_{i=1}^T \tilde w_i}\quad\text{(only over response tokens)}
\]
This preserves the loss scale and keeps \(\beta\) consistent between methods.

Mask/shift note: in causal LM implementations the logits are shifted; weights and masks must align with the labels used for loss (labels shifted by 1). Implementation ensures tokens and weights correspond to the same positions.

---

### 3.2 Methods compared (formulas + implementation notes)

We compare five methods: Uniform (vanilla DPO), Surprisal, CachedGrad, Online Hybrid (TI‑DPO-style adaptation), and Gaussian (control).

#### A. Uniform DPO (baseline)
- \(w_t = 1\).
- Implementation: standard DPO; policy forward + ref forward → per-token advs → sum.

#### B. Surprisal-DPO
- Raw weight:
  \[
  \tilde w_t = -\log\pi_{\text{ref}}(y_t\mid x,y_{<t})
  \]
- Clamp and normalize:
  \[
  w_t = \mathrm{clamp}\bigg(\frac{\tilde w_t}{\frac{1}{T}\sum_i\tilde w_i},\; w_{\min}, w_{\max}\bigg)
  \]
  with \(w_{\min}=0.2,\ w_{\max}=3.0\) (these are practical hyperparameters).
- Implementation: no extra backward; uses ref logprobs already computed for DPO.

#### C. CachedGrad-DPO
- Precompute (offline) per-example importance for chosen and rejected responses using a frozen reference:
  1. Target scalar: response negative log-likelihood under ref
     \[
     L_{\mathrm{NLL}}(x,y) = -\sum_{t} \log\pi_{\text{ref}}(y_t\mid x,y_{<t})
     \]
  2. Gradient attribution: for each token embedding \(e_t\),
     \[
     I_t := \|\nabla_{e_t} L_{\mathrm{NLL}}\|_1
     \]
  3. Normalize to mean 1 → store `chosen_weights` and `rejected_weights`.
- Training: load cached weights aligned to examples and apply them directly at each step (no extra backward during training).
- Tradeoffs: large one-time precompute; training step time ≈ DPO; potential staleness.

#### D. Online Hybrid (approx TI‑DPO)
- Weight blend:
  \[
  \tilde w_t = \lambda \cdot I_{\text{grad},t}^{\text{current}} + (1-\lambda)\cdot P_{\text{gauss},t}
  \]
  with \(\lambda\approx 0.7\).
- \(I_{\text{grad},t}^{\text{current}}=\|\nabla_{e_t}L_{\mathrm{NLL}}\|_1\) computed on the current policy (LoRA on).
- \(P_{\text{gauss},t} = \exp\big(-\tfrac{(t-\mu)^2}{2\sigma^2}\big)\) with \(\mu=(T-1)/2,\ \sigma=T/4\).
- Implementation: recompute \(I_{\text{grad}}\) every \(K\) optimizer steps (we used \(K=10\)), smooth by EMA to avoid instability. This is the most expensive option but closest to the paper’s idea.
- Tradeoffs: extra backward passes (≈ +30–50% wall time); can track importance changes as policy updates.

#### E. Gaussian-only (ablation)
- \(w_t \propto P_{\text{gauss},t}\).
- Implementation: negligible cost; useful ablation to check positional prior value.

---

### 3.5 Evaluation details

- Main metric: held-out preference accuracy (uniform ref-normalized DPO score evaluated on test pairs). For fair comparison we evaluate all trained checkpoints with the same (uniform) scoring unless reporting the diagnostic `weighted_preference_accuracy`.
- Report mean margin \( \mathbb{E}[R(y_w)-R(y_l)] \) as a confidence measure (not directly comparable when evaluation metrics differ).
- Provide approximate 95% CI for accuracy: \(\text{CI} \approx 1.96\sqrt{p(1-p)/n}\) (used to judge whether small gaps are significant).
- Runtime: training step time (mean over 20 measured steps) and one-time precompute cost for CachedGrad.

---

### 3.6 Hyperparameters and practical config

- LoRA: r=8, alpha=16, dropout=0.05, target modules q,k,v,o (or q,v if memory tight).
- β (DPO temperature): 0.1 (keeps sigmoid gradient scale stable across methods).
- Surprisal clamp: [0.2, 3.0]; normalization to mean 1.
- CachedGrad precompute: process whole train split; save per-example weights; include metadata mapping to JSONL order.
- Online hybrid: recompute every 10 optimizer steps; EMA decay 0.9; λ=0.7; clamp final weights to [0.2, 5.0] to avoid spikes.
- Training: ~3k training pairs, 1 epoch, batch 2, gradient accumulation to reach effective batch if needed; total steps ≈187 in our runs.

---

## 3. Main idea and novelty

| Aspect | TI-DPO (paper) | This project |
|--------|----------------|--------------|
| Objective | Token-weighted DPO | Same weighted DPO loss |
| Importance | Online grad + Gaussian | Surprisal, cached grad, online hybrid |
| Triplet loss | Yes | Omitted (compute) |
| Model | (paper setup) | Qwen2.5-0.5B-Instruct, LoRA |
| Reference | Separate ref | Shared weights, LoRA disabled for \(\pi_{\text{ref}}\) |

Novelty: not a new alignment algorithm, but a feasibility study — which cheap importance proxies recover TI-DPO-style benefit vs cost on small models and two preference datasets.

---

## 4. Methods — what we compare and how it works

This is the core of the project. All experiments share one training pipeline; methods differ only in how token weights \(w_t\) are chosen before the same DPO loss is applied.

### 4.1 Standard DPO (baseline)

Idea. Compare chosen vs rejected by the total log-likelihood gap under policy \(\pi_\theta\) relative to a frozen reference \(\pi_{\text{ref}}\):

\[
\Delta_{\text{DPO}} =
\underbrace{\sum_{t \in y_w} \left(\log\pi_\theta - \log\pi_{\text{ref}}\right)}_{\text{chosen score}}
-
\underbrace{\sum_{t \in y_l} \left(\log\pi_\theta - \log\pi_{\text{ref}}\right)}_{\text{rejected score}}
\]

Every response token counts equally (\(w_t = 1\)).

Our implementation (`weight_method: uniform`).
- Load `Qwen2.5-0.5B-Instruct`, attach LoRA → trainable \(\pi_\theta\).
- Reference \(\pi_{\text{ref}}\) = same base weights, LoRA adapters disabled (`model.disable_adapter()`), not a second GPU copy.
- Each training step (`src/trainer.py`):
  1. Tokenize `(prompt, chosen)` and `(prompt, rejected)` with Qwen chat template; mask prompt tokens in `labels` (`-100`).
  2. Forward policy → per-token logprobs on response tokens only.
  3. Forward reference (adapters off) → per-token ref logprobs.
  4. `chosen_score = sum(adv_t)`, same for rejected; `loss = -log σ(β(chosen - rejected))`.
  5. Backprop LoRA only.

Config: `configs/qwen_0.5b_dpo.yaml` → `scripts/04_train.py`.

---

### 4.2 TI-DPO — what the paper adds (and what we keep)

Difference from DPO. TI-DPO uses the same preference pairs and the same sigmoid DPO loss, but replaces uniform sums with weighted sums:

\[
R_w = \sum_t w^w_t \cdot (\log\pi_\theta - \log\pi_{\text{ref}})_t
\quad,\quad
R_l = \sum_t w^l_t \cdot (\log\pi_\theta - \log\pi_{\text{ref}})_t
\]

Token weights come from:

\[
w_t \propto \lambda \cdot I_{\text{grad},t} + (1-\lambda) \cdot P_{\text{gaussian},t}
\]

where \(I_{\text{grad},t} = \|\nabla_{e_t} L\|_1\) (sensitivity of a loss \(L\) to the embedding of token \(t\)), and \(P_{\text{gaussian}}\) favours middle positions.

What TI-DPO also has (we skip).
- Triplet loss on hidden states (anchor = model output, positive = chosen, negative = rejected) — needs extra forwards and generation-like anchors.
- Online recomputation of \(I_{\text{grad}}\) during training — expensive (extra backward through embeddings every step or every few steps).

What we keep from TI-DPO.
- Token-weighted DPO objective (`src/dpo.py`: `weighted_sequence_score` + `compute_dpo_loss_from_logps`).
- Gaussian prior and gradient-based importance as weight sources.
- LoRA fine-tuning on a small instruct model.

What we change (our research question).
- Replace online \(I_{\text{grad}}\) with cached (precompute once) or surprisal (no extra backward).
- Add online hybrid as a costly reference point (recompute \(\lambda\)-blend every 10 steps).
- No triplet term; no separate reward model; no PPO rollouts (`src/ti_ppo/` is PPO code — not used for final DPO experiments).

---

### 4.3 Shared implementation details (all methods)

| Component | Our choice | Code |
|-----------|------------|------|
| Data | JSONL `{prompt, chosen, rejected}` | `data/processed/`, `src/dataset.py` |
| Tokenization | Qwen `apply_chat_template`; prompt masked | `tokenize_preference_pair()` |
| Policy | LoRA on `q,k,v,o` proj, `r=8`, `α=16` | `src/model.py` |
| Reference | Base model, adapters disabled | `get_per_token_logps(..., use_ref=True)` |
| Loss | Weighted DPO, `β=0.1` | `src/dpo.py` |
| Weight norm | Mean weight over response tokens = 1 | `normalize_weights()` |
| Train | ~3k pairs, 1 epoch, batch 2, grad accum 8 | `configs/qwen_0.5b_*.yaml` |

One micro-batch (conceptual):

```
for (x, y_w, y_l) in batch:
    π_θ, π_ref  ← forwards on chosen and rejected
    w_w, w_l    ← method-specific (see below)
    R_w = Σ_t w^w_t · (log π_θ - log π_ref)_t
    R_l = Σ_t w^l_t · (log π_θ - log π_ref)_t
    loss = -log σ(β(R_w - R_l))
    loss.backward()  # LoRA only
```

---

### 4.4 Method 1 — Vanilla DPO (`uniform`)

| | |
|--|--|
| Weights | \(w_t = 1\) on all response tokens |
| vs DPO | This *is* standard DPO |
| vs TI-DPO | No token importance; TI-DPO generalises this with \(w_t\) |
| Extra compute | None |
| Code | `WeightMethod.UNIFORM` → `build_uniform_weights()` |

---

### 4.5 Method 2 — Gaussian-DPO (`gaussian`) — planned, not in final tables

| | |
|--|--|
| Weights | \(\tilde w_t = \exp(-(t-\mu)^2/(2\sigma^2))\), \(\mu=(T-1)/2\), \(\sigma=T/4\), then mean-normalised |
| vs DPO | Same loss, but middle tokens contribute more to \(R_w, R_l\) |
| vs TI-DPO | Uses only \(P_{\text{gaussian}}\), no \(I_{\text{grad}}\) — TI-DPO ablation / positional prior |
| Intuition | Content-free; tests whether position alone helps |
| Extra compute | Negligible (closed form) |
| Code | `build_gaussian_weights()` in `src/dpo.py` |

Skipped in final runs to save GPU time; implementation remains for replication.

---

### 4.6 Method 3 — Surprisal-DPO (`surprisal`)

| | |
|--|--|
| Weights | \(\tilde w_t = -\log\pi_{\text{ref}}(y_t \mid x, y_{<t})\); mean-normalise; clamp to \([0.2, 3.0]\) |
| vs DPO | Tokens the reference finds surprising get larger weight in the preference gradient |
| vs TI-DPO | No gradient attribution — surprisal is a cheap proxy for “informative” tokens; not in the paper’s \(\lambda I_{\text{grad}}\) formula |
| Intuition | Rare / content-heavy tokens may correlate with preference-relevant spans; almost free because ref logprobs are already computed for DPO |
| Extra compute | ~0% (reuses `chosen_ref_logps` / `rejected_ref_logps`) |
| Code | `build_surprisal_weights()`; config `qwen_0.5b_surprisal.yaml` |

Algorithm (each forward):
1. Compute ref logprobs as in vanilla DPO.
2. `w_t = clamp(normalize(-ref_logp_t))`.
3. Plug into weighted DPO loss.

---

### 4.7 Method 4 — CachedGrad-DPO (`cached_grad`) — closest to TI-DPO

| | |
|--|--|
| Weights | \(I_t = \|\nabla_{e_t} \text{NLL}\|_1\) on frozen reference; \(w_t = \text{normalize}(I_t)\) |
| NLL | \(-\sum_{t \in \text{response}} \log\pi_{\text{ref}}(y_t)\) — negative log-likelihood of the full response under ref |
| vs DPO | Same loss shape, fixed importance map per example |
| vs TI-DPO | Same type of \(I_{\text{grad}}\) (embedding-gradient attribution), but: (1) precomputed once before training, not online; (2) no Gaussian blend in training (pure grad cache); (3) computed on initial ref, so weights go stale as LoRA updates \(\pi_\theta\) |
| Extra compute | High once (`scripts/03_precompute_cachedgrad.py`), ~1–3 h for 3k pairs; train step ≈ DPO |
| Code | `src/attribution.py` → `compute_cached_grad_importance()`; weights in `cache/*/train_cachedgrad_weights.pt`; loaded in `PreferenceDataset` |

Precompute (per example, chosen + rejected):
1. Embed `input_ids`; `requires_grad` on embeddings.
2. Forward ref model (`inputs_embeds`); compute response NLL.
3. `backward()` → L1 norm of \(\nabla_e\) per token → normalise.
4. Save list `[{chosen_weights, rejected_weights}, ...]` aligned with JSONL order.

Training: collate loads `chosen_weights` / `rejected_weights` into batch; loss uses them as external \(w_t\). No gradient step for importance during train.

---

### 4.8 Method 5 — Online hybrid (`online_hybrid`) — TI-DPO-style blend, expensive

| | |
|--|--|
| Weights | \(w_t = \text{normalize}\big(\lambda \cdot I_{\text{grad},t} + (1-\lambda) \cdot P_{\text{gaussian},t}\big)\), \(\lambda=0.7\) |
| \(I_{\text{grad}}\) | Same NLL embedding gradient as CachedGrad, but on current policy (with LoRA) |
| Update | Recompute every `importance_update_freq=10` optimizer steps; EMA decay 0.9 per example id |
| vs DPO | Weighted loss + periodic extra backwards |
| vs TI-DPO | Closest to paper’s \(W = \lambda I_{\text{grad}} + (1-\lambda)P_{\text{gaussian}}\); still no triplet; update every 10 steps (not every step) to save time |
| Extra compute | ~1.4× train wall time vs DPO (embedding grad on chosen + rejected) |
| Code | `src/ti_dpo_importance.py` → `OnlineHybridWeightComputer`; config `qwen_0.5b_online_hybrid.yaml` |

Why it exists: upper bound on “faithful TI-DPO importance” cost in our DPO setup — if this does not beat DPO, online full TI-DPO is hard to justify on one GPU.

---

### 4.9 Side-by-side — what actually differs

| | DPO | TI-DPO (paper) | Surprisal (ours) | CachedGrad (ours) | Online hybrid (ours) |
|--|---------|-------------------|----------------------|----------------------|--------------------------|
| Loss | Standard DPO | Weighted DPO | Weighted DPO | Weighted DPO | Weighted DPO |
| \(w_t\) | 1 | \(\lambda I_{\text{grad}} + (1-\lambda)P_{\text{gauss}}\) | \(-\log\pi_{\text{ref}}\) | Cached \(I_{\text{grad}}\) (ref NLL) | Online \(\lambda I_{\text{grad}} + (1-\lambda)P_{\text{gauss}}\) |
| \(I_{\text{grad}}\) timing | — | Online | — | Once, frozen | Every 10 steps |
| Triplet loss | No | Yes | No | No | No |
| Train cost vs DPO | 1× | ≫1× | ~1× | ~1× (+ precompute) | ~1.4× |

---

### 4.10 How we ran the experiments (reproducible workflow)

Per dataset (HH-RLHF or Dahoas/rm-static):

1. Jupyter (`data/dataset.ipynb`): filter lengths, export `train/val/test.jsonl`, report base PrefAcc (~46% on our splits).
2. Train each method: `python scripts/04_train.py --config configs/qwen_0.5b_<method>.yaml`
   - CachedGrad only: `python scripts/03_precompute_cachedgrad.py --split train` first.
3. Evaluate each checkpoint: `python scripts/05_evaluate.py --checkpoint outputs/.../final --split test --exp-name ...`
   - Base row: same script without `--checkpoint` → raw logprob PrefAcc.
4. Aggregate: `python scripts/06_summarize_results.py` → `comparison_table.md`.

Same hyperparameters across methods so differences are only from \(w_t\).

---

### 4.11 Evaluation metrics

| Metric | Definition | When to use |
|--------|------------|-------------|
| Preference accuracy | Fraction of test pairs with \(R(y_w) > R(y_l)\) using ref-normalized DPO scores (uniform weights at eval) | Compare trained methods |
| Base preference accuracy | Fraction where \(\sum_t \log\pi_{\text{base}}(y_w) > \sum_t \log\pi_{\text{base}}(y_l)\) | Untrained model only |
| Mean margin | Average \(R(y_w) - R(y_l)\) | Confidence (not comparable base vs trained) |
| Train `total_train_sec` | Wall time for epoch | `outputs/checkpoints/*/train_stats.json` |
| Precompute sec | CachedGrad only | `cache/*.stats.json` |

All trained models: `evaluation_method: uniform` in config so eval does not use training weights (fair PrefAcc). Code: `src/eval.py`, `scripts/05_evaluate.py`, `scripts/06_summarize_results.py`.

---

## 5. Experiments overview

| Setting | Value |
|---------|--------|
| Model | `Qwen2.5-0.5B-Instruct` |
| Fine-tuning | LoRA `r=8`, `α=16`, targets `q,k,v,o` |
| Train | ~3k pairs, 1 epoch, batch 2, grad accum 8 → 187 optimizer steps |
| Test | 500 pairs |
| Hardware | 1× GPU (V100-class), CUDA 12.1, FP16 |

Datasets:

1. Anthropic HH-RLHF (filtered) — results in `outputs/results/comparison/`
2. Dahoas/rm-static — results in `outputs/ Dahoas_rm-static/results/comparison/`

Data prep: Jupyter → `data/processed/{train,val,test}.jsonl` (see [data/README.md](data/README.md)).

---

## 6. Results

Approximate 95% CI for accuracy: \(\pm 0.043\) on \(n=500\) test pairs.

### 6.1 HH-RLHF (filtered)

| Method | PrefAcc ↑ | Δ vs base | Mean margin | Train sec/step | Total train |
|--------|-----------|-----------|-------------|----------------|-------------|
| Base | 45.6% | — | −4.89 | — | — |
| DPO | 61.6% | +16.0 pp | 1.47 | 0.86 s | ~22 min |
| Surprisal | 63.0% | +17.4 pp | 0.92 | 0.87 s | ~22 min |
| CachedGrad | 62.6% | +17.0 pp | 0.69 | 0.84 s | ~21 min |
| Online hybrid | 61.0% | +15.4 pp | 1.19 | 1.25 s | ~31 min |

Full table: [`outputs/results/comparison/comparison_table.md`](outputs/results/comparison/comparison_table.md)

### 6.2 Dahoas / rm-static

| Method | PrefAcc ↑ | Δ vs base | Mean margin | Runtime sec/step (bench) |
|--------|-----------|-----------|-------------|---------------------------|
| Base | 45.6% | — | −4.89 | — |
| DPO | 61.4% | +15.8 pp | 1.18 | 0.83 s |
| Surprisal | 59.8% | +14.2 pp | 0.61 | 0.80 s |
| CachedGrad | 60.2% | +14.6 pp | 0.67 | 0.93 s |
| Online hybrid | 60.2% | +14.6 pp | 0.59 | 1.40 s |

Full table: [`outputs/ Dahoas_rm-static/results/comparison/comparison_table.md`](outputs/%20Dahoas_rm-static/results/comparison/comparison_table.md)

### 6.3 Efficiency summary

| Method | Extra precompute | Train vs DPO | Quality (HH) | Quality (Dahoas) |
|--------|------------------|--------------|--------------|------------------|
| DPO | None | 1.00× | Good | Best |
| Surprisal | None | ~1.00× | Best | Below DPO |
| CachedGrad | High (once) | ~1.00× | Near best | Below DPO |
| Online hybrid | None | ~1.4–1.5× slower | Below best | Below DPO |

---

## 7. Conclusions

1. DPO + LoRA works on both datasets: base ~46% → ~60–63% test preference accuracy (+15–17 pp).
2. Token weighting is not universally better than vanilla DPO.
   - On HH-RLHF, surprisal and cached grad slightly beat DPO (~1–1.4 pp) — within CI overlap but directionally consistent with token-level safety/helpfulness signal.
   - On rm-static, vanilla DPO wins; all weighted variants are ~1–2 pp lower and margins shrink.
3. Cheap proxies match expensive ones in quality (surprisal ≈ cached grad) but not in fidelity to “online TI-DPO”: online hybrid is slower without consistent gains.
4. CachedGrad does not speed up training vs DPO (by design); savings are vs *hypothetical* online attribution, not vs uniform DPO.
5. Dataset matters more than weighting trick for this model size — HH-RLHF is a better fit for token-importance hypotheses than generic helpfulness pairs.

Practical recommendation: use vanilla DPO on easy helpfulness data; consider surprisal weighting on safety/contrast-heavy data if you need a cheap token-aware variant.

---

## 8. Weaknesses and limitations

1. No official TI-DPO release — hybrid online weights approximate the paper; triplet loss omitted.
2. Single model size (0.5B); 1.5B experiments not completed in final tables.
3. CachedGrad staleness — weights fixed at pretrain/ref; policy drifts under LoRA.
4. Statistical power — \(n=500\), CI ≈ ±4.3 pp; small method gaps may be noise.
5. PrefAcc ≠ human quality — no generation eval or RM scoring in final numbers.
6. Base vs trained metrics — base uses raw log-prob; trained uses ref-normalized DPO score (PrefAcc still interpretable).
7. Gaussian baseline not run in final comparison.

### Code walkthrough (for 2-point algo slide)

```
scripts/04_train.py
  └─ src/trainer.py          # loop: forward π_θ and π_ref, get weights, backprop LoRA only
       └─ src/dpo.py          # per-token logps, weighted DPO loss
            ├─ uniform / gaussian / surprisal   (on-the-fly weights)
            ├─ cached_grad    (weights from cache/*.pt)
            └─ online_hybrid  (src/ti_dpo_importance.py)
scripts/05_evaluate.py         # PrefAcc, margins, timing
scripts/06_summarize_results.py  # comparison_table.md
```

Key line: weighted scores `R_w`, `R_l` → `L = -log σ(β(R_w - R_l))` in `compute_dpo_loss_from_logps()`.

---
