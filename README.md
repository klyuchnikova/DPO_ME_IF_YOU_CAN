# DPO Me If You Can

**DPO Me If You Can** — efficient token-weighted DPO for `Qwen2.5-0.5B-Instruct` under limited compute.

We study whether cheap token-importance approximations can match the *spirit* of **TI-DPO** (Token-Importance Guided Direct Preference Optimization) without online gradient attribution at every training step. Full TI-DPO code is not public; this repository is our own replication and ablation.

---

## 1. Motivation

Standard DPO optimizes a **sequence-level** preference objective: every response token contributes equally to

\[
\Delta_{\text{DPO}} =
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\quad,\quad
L_{\text{DPO}} = -\log \sigma(\beta \Delta_{\text{DPO}})
\]

TI-DPO argues that some tokens matter more for human preference (e.g. safety phrases, key facts) than filler text. It replaces the sum with **token weights** \(w_t\):

\[
\Delta_{\text{weighted}} =
\sum_t w^w_t \left[\log \pi_\theta(y^w_t|\cdot) - \log \pi_{\text{ref}}(y^w_t|\cdot)\right]
-
\sum_t w^l_t \left[\log \pi_\theta(y^l_t|\cdot) - \log \pi_{\text{ref}}(y^l_t|\cdot)\right]
\]

In the paper, \(w_t\) combines **gradient attribution** and a **Gaussian positional prior**:

\[
W = \lambda I_{\text{grad}} + (1-\lambda) P_{\text{gaussian}}
\]

Computing \(I_{\text{grad}}\) online (backward through input embeddings every step) is expensive on a single GPU.

**Our goal**:

Make token-importance DPO cheaper for small LLMs by replacing online gradient attribution with **cached** or **approximate** token-importance weights — while keeping LoRA DPO training practical on one V100.

We **do not** reproduce TI-DPO's triplet loss or full PPO pipeline.

**Research questions:**

1. Can token-weighted DPO improve over vanilla DPO on small LLMs?
2. Can surprisal (\(-\log \pi_{\text{ref}}\)) work as a almost-free proxy?
3. Can cached gradients match online attribution at near-DPO train speed?
4. Does the answer depend on the dataset (general helpfulness vs safety-style preferences)?

---

## 2. Place in the literature

<!-- TODO: Add related work for the presentation/report.
Suggested entries:
- Rafailov et al., DPO (2023)
- TI-DPO paper (token-importance guided DPO) — cite when available
- LoRA / PEFT for efficient fine-tuning
- Reward-model / RLHF baselines (contrast only)
-->

---

## 3. Main idea and novelty

| Aspect | TI-DPO (paper) | This project |
|--------|----------------|--------------|
| Objective | Token-weighted DPO | Same weighted DPO loss |
| Importance | Online grad + Gaussian | **Surprisal**, **cached grad**, **online hybrid** |
| Triplet loss | Yes | **Omitted** (compute) |
| Model | (paper setup) | **Qwen2.5-0.5B-Instruct**, LoRA |
| Reference | Separate ref | **Shared weights**, LoRA disabled for \(\pi_{\text{ref}}\) |

**Novelty:** not a new alignment algorithm, but a **feasibility study** — which cheap importance proxies recover TI-DPO-style benefit vs cost on small models and two preference datasets.

---

## 4. Methods — what we compare and how it works

This is the core of the project. All experiments share **one training pipeline**; methods differ only in **how token weights \(w_t\) are chosen** before the same DPO loss is applied.

### 4.1 Standard DPO (baseline)

**Idea.** Compare chosen vs rejected by the **total** log-likelihood gap under policy \(\pi_\theta\) relative to a frozen reference \(\pi_{\text{ref}}\):

\[
\Delta_{\text{DPO}} =
\underbrace{\sum_{t \in y_w} \left(\log\pi_\theta - \log\pi_{\text{ref}}\right)}_{\text{chosen score}}
-
\underbrace{\sum_{t \in y_l} \left(\log\pi_\theta - \log\pi_{\text{ref}}\right)}_{\text{rejected score}}
\]

Every response token counts equally (\(w_t = 1\)).

**Our implementation (`weight_method: uniform`).**
- Load `Qwen2.5-0.5B-Instruct`, attach LoRA → trainable \(\pi_\theta\).
- Reference \(\pi_{\text{ref}}\) = **same base weights**, LoRA adapters **disabled** (`model.disable_adapter()`), not a second GPU copy.
- Each training step (`src/trainer.py`):
  1. Tokenize `(prompt, chosen)` and `(prompt, rejected)` with Qwen chat template; mask prompt tokens in `labels` (`-100`).
  2. Forward policy → per-token logprobs on **response tokens only**.
  3. Forward reference (adapters off) → per-token ref logprobs.
  4. `chosen_score = sum(adv_t)`, same for rejected; `loss = -log σ(β(chosen - rejected))`.
  5. Backprop **LoRA only**.

**Config:** `configs/qwen_0.5b_dpo.yaml` → `scripts/04_train.py`.

---

### 4.2 TI-DPO — what the paper adds (and what we keep)

**Difference from DPO.** TI-DPO uses the **same preference pairs** and the **same sigmoid DPO loss**, but replaces uniform sums with **weighted** sums:

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

**What TI-DPO also has (we skip).**
- **Triplet loss** on hidden states (anchor = model output, positive = chosen, negative = rejected) — needs extra forwards and generation-like anchors.
- **Online** recomputation of \(I_{\text{grad}}\) during training — expensive (extra backward through embeddings every step or every few steps).

**What we keep from TI-DPO.**
- Token-weighted DPO objective (`src/dpo.py`: `weighted_sequence_score` + `compute_dpo_loss_from_logps`).
- Gaussian prior and gradient-based importance as **weight sources**.
- LoRA fine-tuning on a small instruct model.

**What we change (our research question).**
- Replace **online** \(I_{\text{grad}}\) with **cached** (precompute once) or **surprisal** (no extra backward).
- Add **online hybrid** as a costly reference point (recompute \(\lambda\)-blend every 10 steps).
- No triplet term; no separate reward model; no PPO rollouts (`src/ti_ppo/` is PPO code — **not** used for final DPO experiments).

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

**One micro-batch (conceptual):**

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
| **Weights** | \(w_t = 1\) on all response tokens |
| **vs DPO** | This *is* standard DPO |
| **vs TI-DPO** | No token importance; TI-DPO generalises this with \(w_t\) |
| **Extra compute** | None |
| **Code** | `WeightMethod.UNIFORM` → `build_uniform_weights()` |

---

### 4.5 Method 2 — Gaussian-DPO (`gaussian`) — planned, not in final tables

| | |
|--|--|
| **Weights** | \(\tilde w_t = \exp(-(t-\mu)^2/(2\sigma^2))\), \(\mu=(T-1)/2\), \(\sigma=T/4\), then mean-normalised |
| **vs DPO** | Same loss, but middle tokens contribute more to \(R_w, R_l\) |
| **vs TI-DPO** | Uses only \(P_{\text{gaussian}}\), no \(I_{\text{grad}}\) — TI-DPO ablation / positional prior |
| **Intuition** | Content-free; tests whether position alone helps |
| **Extra compute** | Negligible (closed form) |
| **Code** | `build_gaussian_weights()` in `src/dpo.py` |

Skipped in final runs to save GPU time; implementation remains for replication.

---

### 4.6 Method 3 — Surprisal-DPO (`surprisal`)

| | |
|--|--|
| **Weights** | \(\tilde w_t = -\log\pi_{\text{ref}}(y_t \mid x, y_{<t})\); mean-normalise; clamp to \([0.2, 3.0]\) |
| **vs DPO** | Tokens the reference finds **surprising** get larger weight in the preference gradient |
| **vs TI-DPO** | **No gradient attribution** — surprisal is a cheap proxy for “informative” tokens; not in the paper’s \(\lambda I_{\text{grad}}\) formula |
| **Intuition** | Rare / content-heavy tokens may correlate with preference-relevant spans; almost free because ref logprobs are already computed for DPO |
| **Extra compute** | ~0% (reuses `chosen_ref_logps` / `rejected_ref_logps`) |
| **Code** | `build_surprisal_weights()`; config `qwen_0.5b_surprisal.yaml` |

**Algorithm (each forward):**
1. Compute ref logprobs as in vanilla DPO.
2. `w_t = clamp(normalize(-ref_logp_t))`.
3. Plug into weighted DPO loss.

---

### 4.7 Method 4 — CachedGrad-DPO (`cached_grad`) — closest to TI-DPO

| | |
|--|--|
| **Weights** | \(I_t = \|\nabla_{e_t} \text{NLL}\|_1\) on **frozen reference**; \(w_t = \text{normalize}(I_t)\) |
| **NLL** | \(-\sum_{t \in \text{response}} \log\pi_{\text{ref}}(y_t)\) — negative log-likelihood of the full response under ref |
| **vs DPO** | Same loss shape, fixed importance map per example |
| **vs TI-DPO** | Same **type** of \(I_{\text{grad}}\) (embedding-gradient attribution), but: (1) **precomputed once** before training, not online; (2) **no Gaussian blend** in training (pure grad cache); (3) computed on **initial ref**, so weights go **stale** as LoRA updates \(\pi_\theta\) |
| **Extra compute** | **High once** (`scripts/03_precompute_cachedgrad.py`), ~1–3 h for 3k pairs; train step ≈ DPO |
| **Code** | `src/attribution.py` → `compute_cached_grad_importance()`; weights in `cache/*/train_cachedgrad_weights.pt`; loaded in `PreferenceDataset` |

**Precompute (per example, chosen + rejected):**
1. Embed `input_ids`; `requires_grad` on embeddings.
2. Forward ref model (`inputs_embeds`); compute response NLL.
3. `backward()` → L1 norm of \(\nabla_e\) per token → normalise.
4. Save list `[{chosen_weights, rejected_weights}, ...]` aligned with JSONL order.

**Training:** collate loads `chosen_weights` / `rejected_weights` into batch; loss uses them as external \(w_t\). No gradient step for importance during train.

---

### 4.8 Method 5 — Online hybrid (`online_hybrid`) — TI-DPO-style blend, expensive

| | |
|--|--|
| **Weights** | \(w_t = \text{normalize}\big(\lambda \cdot I_{\text{grad},t} + (1-\lambda) \cdot P_{\text{gaussian},t}\big)\), \(\lambda=0.7\) |
| **\(I_{\text{grad}}\)** | Same NLL embedding gradient as CachedGrad, but on **current** policy (with LoRA) |
| **Update** | Recompute every `importance_update_freq=10` **optimizer steps**; EMA decay 0.9 per example id |
| **vs DPO** | Weighted loss + periodic extra backwards |
| **vs TI-DPO** | Closest to paper’s \(W = \lambda I_{\text{grad}} + (1-\lambda)P_{\text{gaussian}}\); still **no triplet**; update every 10 steps (not every step) to save time |
| **Extra compute** | ~**1.4×** train wall time vs DPO (embedding grad on chosen + rejected) |
| **Code** | `src/ti_dpo_importance.py` → `OnlineHybridWeightComputer`; config `qwen_0.5b_online_hybrid.yaml` |

**Why it exists:** upper bound on “faithful TI-DPO importance” cost in our DPO setup — if this does not beat DPO, online full TI-DPO is hard to justify on one GPU.

---

### 4.9 Side-by-side — what actually differs

| | **DPO** | **TI-DPO (paper)** | **Surprisal (ours)** | **CachedGrad (ours)** | **Online hybrid (ours)** |
|--|---------|-------------------|----------------------|----------------------|--------------------------|
| Loss | Standard DPO | Weighted DPO | Weighted DPO | Weighted DPO | Weighted DPO |
| \(w_t\) | 1 | \(\lambda I_{\text{grad}} + (1-\lambda)P_{\text{gauss}}\) | \(-\log\pi_{\text{ref}}\) | Cached \(I_{\text{grad}}\) (ref NLL) | Online \(\lambda I_{\text{grad}} + (1-\lambda)P_{\text{gauss}}\) |
| \(I_{\text{grad}}\) timing | — | Online | — | Once, frozen | Every 10 steps |
| Triplet loss | No | Yes | No | No | No |
| Train cost vs DPO | 1× | ≫1× | ~1× | ~1× (+ precompute) | ~1.4× |

---

### 4.10 How we ran the experiments (reproducible workflow)

**Per dataset** (HH-RLHF or Dahoas/rm-static):

1. **Jupyter** (`data/dataset.ipynb`): filter lengths, export `train/val/test.jsonl`, report base PrefAcc (~46% on our splits).
2. **Train** each method: `python scripts/04_train.py --config configs/qwen_0.5b_<method>.yaml`
   - CachedGrad only: `python scripts/03_precompute_cachedgrad.py --split train` first.
3. **Evaluate** each checkpoint: `python scripts/05_evaluate.py --checkpoint outputs/.../final --split test --exp-name ...`
   - Base row: same script **without** `--checkpoint` → raw logprob PrefAcc.
4. **Aggregate**: `python scripts/06_summarize_results.py` → `comparison_table.md`.

Same hyperparameters across methods so differences are **only** from \(w_t\).

---

### 4.11 Evaluation metrics

| Metric | Definition | When to use |
|--------|------------|-------------|
| **Preference accuracy** | Fraction of test pairs with \(R(y_w) > R(y_l)\) using **ref-normalized** DPO scores (uniform weights at eval) | Compare **trained** methods |
| **Base preference accuracy** | Fraction where \(\sum_t \log\pi_{\text{base}}(y_w) > \sum_t \log\pi_{\text{base}}(y_l)\) | Untrained model only |
| **Mean margin** | Average \(R(y_w) - R(y_l)\) | Confidence (not comparable base vs trained) |
| **Train `total_train_sec`** | Wall time for epoch | `outputs/checkpoints/*/train_stats.json` |
| **Precompute sec** | CachedGrad only | `cache/*.stats.json` |

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

**Datasets:**

1. **Anthropic HH-RLHF** (filtered) — results in `outputs/results/comparison/`
2. **Dahoas/rm-static** — results in `outputs/ Dahoas_rm-static/results/comparison/`

Data prep: Jupyter → `data/processed/{train,val,test}.jsonl` (see [data/README.md](data/README.md)).

---

## 6. Results

Approximate 95% CI for accuracy: \(\pm 0.043\) on \(n=500\) test pairs.

### 6.1 HH-RLHF (filtered)

| Method | PrefAcc ↑ | Δ vs base | Mean margin | Train sec/step | Total train |
|--------|-----------|-----------|-------------|----------------|-------------|
| **Base** | 45.6% | — | −4.89 | — | — |
| DPO | 61.6% | +16.0 pp | 1.47 | 0.86 s | ~22 min |
| **Surprisal** | **63.0%** | **+17.4 pp** | 0.92 | 0.87 s | ~22 min |
| CachedGrad | 62.6% | +17.0 pp | 0.69 | 0.84 s | ~21 min |
| Online hybrid | 61.0% | +15.4 pp | 1.19 | **1.25 s** | **~31 min** |

Full table: [`outputs/results/comparison/comparison_table.md`](outputs/results/comparison/comparison_table.md)

### 6.2 Dahoas / rm-static

| Method | PrefAcc ↑ | Δ vs base | Mean margin | Runtime sec/step (bench) |
|--------|-----------|-----------|-------------|---------------------------|
| **Base** | 45.6% | — | −4.89 | — |
| **DPO** | **61.4%** | **+15.8 pp** | **1.18** | 0.83 s |
| Surprisal | 59.8% | +14.2 pp | 0.61 | 0.80 s |
| CachedGrad | 60.2% | +14.6 pp | 0.67 | 0.93 s |
| Online hybrid | 60.2% | +14.6 pp | 0.59 | 1.40 s |

Full table: [`outputs/ Dahoas_rm-static/results/comparison/comparison_table.md`](outputs/%20Dahoas_rm-static/results/comparison/comparison_table.md)

### 6.3 Efficiency summary

| Method | Extra precompute | Train vs DPO | Quality (HH) | Quality (Dahoas) |
|--------|------------------|--------------|--------------|------------------|
| DPO | None | 1.00× | Good | **Best** |
| Surprisal | None | ~1.00× | **Best** | Below DPO |
| CachedGrad | High (once) | ~1.00× | Near best | Below DPO |
| Online hybrid | None | **~1.4–1.5× slower** | Below best | Below DPO |

---

## 7. Conclusions

1. **DPO + LoRA works** on both datasets: base ~46% → **~60–63%** test preference accuracy (+15–17 pp).
2. **Token weighting is not universally better than vanilla DPO.**
   - On **HH-RLHF**, surprisal and cached grad slightly **beat** DPO (~1–1.4 pp) — within CI overlap but directionally consistent with token-level safety/helpfulness signal.
   - On **rm-static**, **vanilla DPO wins**; all weighted variants are ~1–2 pp lower and margins shrink.
3. **Cheap proxies match expensive ones in quality** (surprisal ≈ cached grad) but **not** in fidelity to “online TI-DPO”: online hybrid is slower without consistent gains.
4. **CachedGrad does not speed up training** vs DPO (by design); savings are vs *hypothetical* online attribution, not vs uniform DPO.
5. **Dataset matters more than weighting trick** for this model size — HH-RLHF is a better fit for token-importance hypotheses than generic helpfulness pairs.

**Practical recommendation:** use **vanilla DPO** on easy helpfulness data; consider **surprisal weighting** on safety/contrast-heavy data if you need a cheap token-aware variant.

---

## 8. Weaknesses and limitations

1. **No official TI-DPO release** — hybrid online weights approximate the paper; triplet loss omitted.
2. **Single model size** (0.5B); 1.5B experiments not completed in final tables.
3. **CachedGrad staleness** — weights fixed at pretrain/ref; policy drifts under LoRA.
4. **Statistical power** — \(n=500\), CI ≈ ±4.3 pp; small method gaps may be noise.
5. **PrefAcc ≠ human quality** — no generation eval or RM scoring in final numbers.
6. **Base vs trained metrics** — base uses raw log-prob; trained uses ref-normalized DPO score (PrefAcc still interpretable).
7. **Gaussian baseline** not run in final comparison.

---

## 9. Presentation plan (course rubric)

Original plan mapping — use this as your slide outline:

| Rubric item | Points | Where in this repo / README |
|-------------|--------|-----------------------------|
| **1) Motivation** | basic | [§1 Motivation](#1-motivation) |
| **2) Place in the literature** | basic | [§2 Literature](#2-place-in-the-literature) *(fill in)* |
| **3) Main idea and novelty** | basic | [§3 Main idea](#3-main-idea-and-novelty) |
| **4) Experiments overview** | basic | [§5 Experiments](#5-experiments-overview) |
| **5) Weaknesses** | basic | [§8 Weaknesses](#8-weaknesses-and-limitations) |
| **Code explanation of main algo** | 2 pts | [§4 Methods](#4-methods--what-we-compare-and-how-it-works) |
| **Own ablation / replication** | 2 pts | [§6 Results](#6-results), two datasets, five methods |

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

**Key line:** weighted scores `R_w`, `R_l` → `L = -log σ(β(R_w - R_l))` in `compute_dpo_loss_from_logps()`.

---

## 10. Reproduction (short)

```bash
pip install -r requirements.txt

# Train
python scripts/04_train.py --config configs/qwen_0.5b_dpo.yaml
python scripts/04_train.py --config configs/qwen_0.5b_surprisal.yaml
python scripts/03_precompute_cachedgrad.py --config configs/qwen_0.5b_cachedgrad.yaml --split train
python scripts/04_train.py --config configs/qwen_0.5b_cachedgrad.yaml

# Eval + aggregate
python scripts/05_evaluate.py --config configs/qwen_0.5b_dpo.yaml --exp-name qwen_0.5b_base --split test --skip-runtime --skip-visualize
python scripts/05_evaluate.py --config configs/qwen_0.5b_dpo.yaml --checkpoint outputs/checkpoints/qwen_0.5b_dpo/final --exp-name qwen_0.5b_dpo --split test
python scripts/06_summarize_results.py --results-root outputs/results --split test
```

Project layout: `configs/`, `src/`, `scripts/`, `data/processed/`, `cache/`, `outputs/`.

See [Idea.md](Idea.md) for the full project design notes.
