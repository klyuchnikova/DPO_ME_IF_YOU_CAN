# Cumulative experiment report

## Filters

- Results root: `outputs/results`
- Split: `test`
- Model filter: `None`
- Number of rows: `5`

## Notes

- Warning: multiple evaluation methods appear in the table: , raw_logprob, uniform. Direct accuracy comparisons are safest only when evaluation methods match.
- Base rows use raw response log-probability, while trained DPO rows may use reference-normalized margins. Preference accuracy can still be useful, but margins/losses are not directly comparable between base and DPO-style rows.

## Best method by model

| model | best_method | best_pref_acc | delta_vs_base | mean_margin | total_wall_sec | runtime_step_sec | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | dpo | 0.6140 | 0.1580 | 1.1818 |  | 0.83s | qwen_0.5b_dpo |

## Compact comparison table

| model_short | method | preference_accuracy | preference_accuracy_ci95 | delta_vs_base | mean_margin | median_margin | mean_loss | global_step | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | num_examples | evaluation_method | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | base | 0.4560 | 0.0437 | 0.0000 | -4.8909 | -5.0309 | 0.0000 | 187 | 0.87s |  |  |  |  | 500 | raw_logprob | qwen_0.5b_base |
| Qwen2.5-0.5B-Instruct | dpo | 0.6140 | 0.0427 | 0.1580 | 1.1818 | 0.9239 | 0.6628 |  |  | 0.83s |  |  |  | 500 |  | qwen_0.5b_dpo |
| Qwen2.5-0.5B-Instruct | surprisal | 0.5980 | 0.0430 | 0.1420 | 0.6116 | 0.4880 | 0.6625 |  |  | 0.80s |  |  |  | 500 |  | qwen_0.5b_surprisal |
| Qwen2.5-0.5B-Instruct | cached_grad | 0.6020 | 0.0429 | 0.1460 | 0.6679 | 0.5589 | 0.6719 |  |  | 0.93s |  |  |  | 500 | uniform | qwen_0.5b_cachedgrad |
| Qwen2.5-0.5B-Instruct | online_hybrid | 0.6020 | 0.0429 | 0.1460 | 0.5875 | 0.5281 | 0.6709 |  |  | 1.40s |  |  |  | 500 | uniform | qwen_0.5b_online_hybrid |

## Per-model tables

### Qwen/Qwen2.5-0.5B-Instruct

| model_short | method | preference_accuracy | preference_accuracy_ci95 | delta_vs_base | mean_margin | median_margin | mean_loss | global_step | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | num_examples | evaluation_method | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | base | 0.4560 | 0.0437 | 0.0000 | -4.8909 | -5.0309 | 0.0000 | 187 | 0.87s |  |  |  |  | 500 | raw_logprob | qwen_0.5b_base |
| Qwen2.5-0.5B-Instruct | dpo | 0.6140 | 0.0427 | 0.1580 | 1.1818 | 0.9239 | 0.6628 |  |  | 0.83s |  |  |  | 500 |  | qwen_0.5b_dpo |
| Qwen2.5-0.5B-Instruct | surprisal | 0.5980 | 0.0430 | 0.1420 | 0.6116 | 0.4880 | 0.6625 |  |  | 0.80s |  |  |  | 500 |  | qwen_0.5b_surprisal |
| Qwen2.5-0.5B-Instruct | cached_grad | 0.6020 | 0.0429 | 0.1460 | 0.6679 | 0.5589 | 0.6719 |  |  | 0.93s |  |  |  | 500 | uniform | qwen_0.5b_cachedgrad |
| Qwen2.5-0.5B-Instruct | online_hybrid | 0.6020 | 0.0429 | 0.1460 | 0.5875 | 0.5281 | 0.6709 |  |  | 1.40s |  |  |  | 500 | uniform | qwen_0.5b_online_hybrid |

## Full table

| exp_name | model_name | method | training_method | weight_method | evaluation_method | checkpoint | split | num_examples | preference_accuracy | preference_accuracy_ci95 | weighted_preference_accuracy | base_preference_accuracy | delta_vs_base | relative_delta_vs_base | mean_margin | margin_delta_vs_base | median_margin | mean_loss | global_step | train_mean_loss | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | metric_note | summary_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_0.5b_base | Qwen/Qwen2.5-0.5B-Instruct | base | base | uniform | raw_logprob | base | test | 500 | 0.4560 | 0.0437 |  | 0.4560 | 0.0000 | 0.0000 | -4.8909 | 0.0000 | -5.0309 | 0.0000 | 187 | 0.6693 | 0.87s |  |  |  |  | sum of response log-probs; not ref-normalized | outputs/results/qwen_0.5b_base/summary.json |
| qwen_0.5b_dpo | Qwen/Qwen2.5-0.5B-Instruct | dpo | dpo | uniform |  | outputs/checkpoints/qwen_0.5b_dpo/final | test | 500 | 0.6140 | 0.0427 | 0.6140 | 0.4560 | 0.1580 | 0.3465 | 1.1818 | 6.0727 | 0.9239 | 0.6628 |  |  |  | 0.83s |  |  |  |  | outputs/results/qwen_0.5b_dpo/summary.json |
| qwen_0.5b_surprisal | Qwen/Qwen2.5-0.5B-Instruct | surprisal | surprisal | surprisal |  | outputs/checkpoints/qwen_0.5b_surprisal/final | test | 500 | 0.5980 | 0.0430 | 0.6000 | 0.4560 | 0.1420 | 0.3114 | 0.6116 | 5.5024 | 0.4880 | 0.6625 |  |  |  | 0.80s |  |  |  |  | outputs/results/qwen_0.5b_surprisal/summary.json |
| qwen_0.5b_cachedgrad | Qwen/Qwen2.5-0.5B-Instruct | cached_grad | cached_grad | cached_grad | uniform | outputs/checkpoints/qwen_0.5b_cachedgrad/final | test | 500 | 0.6020 | 0.0429 | 0.6020 | 0.4560 | 0.1460 | 0.3202 | 0.6679 | 5.5588 | 0.5589 | 0.6719 |  |  |  | 0.93s |  |  |  |  | outputs/results/qwen_0.5b_cachedgrad/summary.json |
| qwen_0.5b_online_hybrid | Qwen/Qwen2.5-0.5B-Instruct | online_hybrid | online_hybrid | online_hybrid | uniform | outputs/checkpoints/qwen_0.5b_online_hybrid/final | test | 500 | 0.6020 | 0.0429 | 0.6020 | 0.4560 | 0.1460 | 0.3202 | 0.5875 | 5.4784 | 0.5281 | 0.6709 |  |  |  | 1.40s |  |  |  |  | outputs/results/qwen_0.5b_online_hybrid/summary.json |

