# Cumulative experiment report

## Filters

- Results root: `outputs/results`
- Split: `test`
- Model filter: `None`
- Number of rows: `5`

## Notes

- Warning: multiple evaluation methods appear in the table: raw_logprob, uniform. Direct accuracy comparisons are safest only when evaluation methods match.
- Base rows use raw response log-probability, while trained DPO rows may use reference-normalized margins. Preference accuracy can still be useful, but margins/losses are not directly comparable between base and DPO-style rows.

## Best method by model

| model | best_method | best_pref_acc | delta_vs_base | mean_margin | total_wall_sec | runtime_step_sec | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | surprisal | 0.6300 | 0.1740 | 0.9215 | 21.89m | 0.79s | qwen_0.5b_surprisal |

## Compact comparison table

| model_short | method | preference_accuracy | preference_accuracy_ci95 | delta_vs_base | mean_margin | median_margin | mean_loss | global_step | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | num_examples | evaluation_method | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | base | 0.4560 | 0.0437 | 0.0000 | -4.8909 | -5.0309 | 0.0000 | 187 | 0.86s |  |  | 21.84m | 21.84m | 500 | raw_logprob | qwen_0.5b_base |
| Qwen2.5-0.5B-Instruct | dpo | 0.6160 | 0.0426 | 0.1600 | 1.4709 | 1.1919 | 0.6504 | 187 | 0.86s | 0.78s |  | 21.84m | 21.84m | 500 | uniform | qwen_0.5b_dpo |
| Qwen2.5-0.5B-Instruct | surprisal | 0.6300 | 0.0423 | 0.1740 | 0.9215 | 0.7195 | 0.6580 | 187 | 0.87s | 0.79s |  | 21.89m | 21.89m | 500 | uniform | qwen_0.5b_surprisal |
| Qwen2.5-0.5B-Instruct | cached_grad | 0.6260 | 0.0424 | 0.1700 | 0.6903 | 0.4342 | 0.6676 | 187 | 0.84s | 0.80s |  | 21.33m | 21.33m | 500 | uniform | qwen_0.5b_cachedgrad |
| Qwen2.5-0.5B-Instruct | online_hybrid | 0.6100 | 0.0428 | 0.1540 | 1.1922 | 0.9277 | 0.6539 | 187 | 1.25s | 0.89s |  | 31.37m | 31.37m | 500 | uniform | qwen_0.5b_online_hybrid |

## Per-model tables

### Qwen/Qwen2.5-0.5B-Instruct

| model_short | method | preference_accuracy | preference_accuracy_ci95 | delta_vs_base | mean_margin | median_margin | mean_loss | global_step | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | num_examples | evaluation_method | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | base | 0.4560 | 0.0437 | 0.0000 | -4.8909 | -5.0309 | 0.0000 | 187 | 0.86s |  |  | 21.84m | 21.84m | 500 | raw_logprob | qwen_0.5b_base |
| Qwen2.5-0.5B-Instruct | dpo | 0.6160 | 0.0426 | 0.1600 | 1.4709 | 1.1919 | 0.6504 | 187 | 0.86s | 0.78s |  | 21.84m | 21.84m | 500 | uniform | qwen_0.5b_dpo |
| Qwen2.5-0.5B-Instruct | surprisal | 0.6300 | 0.0423 | 0.1740 | 0.9215 | 0.7195 | 0.6580 | 187 | 0.87s | 0.79s |  | 21.89m | 21.89m | 500 | uniform | qwen_0.5b_surprisal |
| Qwen2.5-0.5B-Instruct | cached_grad | 0.6260 | 0.0424 | 0.1700 | 0.6903 | 0.4342 | 0.6676 | 187 | 0.84s | 0.80s |  | 21.33m | 21.33m | 500 | uniform | qwen_0.5b_cachedgrad |
| Qwen2.5-0.5B-Instruct | online_hybrid | 0.6100 | 0.0428 | 0.1540 | 1.1922 | 0.9277 | 0.6539 | 187 | 1.25s | 0.89s |  | 31.37m | 31.37m | 500 | uniform | qwen_0.5b_online_hybrid |

## Full table

| exp_name | model_name | method | training_method | weight_method | evaluation_method | checkpoint | split | num_examples | preference_accuracy | preference_accuracy_ci95 | weighted_preference_accuracy | base_preference_accuracy | delta_vs_base | relative_delta_vs_base | mean_margin | margin_delta_vs_base | median_margin | mean_loss | global_step | train_mean_loss | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | metric_note | summary_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_0.5b_base | Qwen/Qwen2.5-0.5B-Instruct | base | base | uniform | raw_logprob | base | test | 500 | 0.4560 | 0.0437 |  | 0.4560 | 0.0000 | 0.0000 | -4.8909 | 0.0000 | -5.0309 | 0.0000 | 187 | 0.6610 | 0.86s |  |  | 21.84m | 21.84m | sum of response log-probs; not ref-normalized | outputs/results/qwen_0.5b_base/summary.json |
| qwen_0.5b_dpo | Qwen/Qwen2.5-0.5B-Instruct | dpo | uniform | uniform | uniform | outputs/checkpoints/qwen_0.5b_dpo/final | test | 500 | 0.6160 | 0.0426 | 0.6160 | 0.4560 | 0.1600 | 0.3509 | 1.4709 | 6.3617 | 1.1919 | 0.6504 | 187 | 0.6610 | 0.86s | 0.78s |  | 21.84m | 21.84m |  | outputs/results/qwen_0.5b_dpo/summary.json |
| qwen_0.5b_surprisal | Qwen/Qwen2.5-0.5B-Instruct | surprisal | surprisal | surprisal | uniform | outputs/checkpoints/qwen_0.5b_surprisal/final | test | 500 | 0.6300 | 0.0423 | 0.6300 | 0.4560 | 0.1740 | 0.3816 | 0.9215 | 5.8124 | 0.7195 | 0.6580 | 187 | 0.6571 | 0.87s | 0.79s |  | 21.89m | 21.89m |  | outputs/results/qwen_0.5b_surprisal/summary.json |
| qwen_0.5b_cachedgrad | Qwen/Qwen2.5-0.5B-Instruct | cached_grad | cached_grad | cached_grad | uniform | outputs/checkpoints/qwen_0.5b_cachedgrad/final | test | 500 | 0.6260 | 0.0424 | 0.6260 | 0.4560 | 0.1700 | 0.3728 | 0.6903 | 5.5812 | 0.4342 | 0.6676 | 187 | 0.6905 | 0.84s | 0.80s |  | 21.33m | 21.33m |  | outputs/results/qwen_0.5b_cachedgrad/summary.json |
| qwen_0.5b_online_hybrid | Qwen/Qwen2.5-0.5B-Instruct | online_hybrid | online_hybrid | online_hybrid | uniform | outputs/checkpoints/qwen_0.5b_online_hybrid/final | test | 500 | 0.6100 | 0.0428 | 0.6100 | 0.4560 | 0.1540 | 0.3377 | 1.1922 | 6.0831 | 0.9277 | 0.6539 | 187 | 0.6653 | 1.25s | 0.89s |  | 31.37m | 31.37m |  | outputs/results/qwen_0.5b_online_hybrid/summary.json |

