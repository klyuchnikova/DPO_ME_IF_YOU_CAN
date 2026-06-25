# Preference comparison

| model_short | method | preference_accuracy | preference_accuracy_ci95 | delta_vs_base | mean_margin | median_margin | mean_loss | global_step | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | num_examples | evaluation_method | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | base | 0.4560 | 0.0437 | 0.0000 | -4.8909 | -5.0309 | 0.0000 | 187 | 0.87s |  |  |  |  | 500 | raw_logprob | qwen_0.5b_base |
| Qwen2.5-0.5B-Instruct | dpo | 0.6140 | 0.0427 | 0.1580 | 1.1818 | 0.9239 | 0.6628 |  |  | 0.83s |  |  |  | 500 |  | qwen_0.5b_dpo |
| Qwen2.5-0.5B-Instruct | surprisal | 0.5980 | 0.0430 | 0.1420 | 0.6116 | 0.4880 | 0.6625 |  |  | 0.80s |  |  |  | 500 |  | qwen_0.5b_surprisal |
| Qwen2.5-0.5B-Instruct | cached_grad | 0.6020 | 0.0429 | 0.1460 | 0.6679 | 0.5589 | 0.6719 |  |  | 0.93s |  |  |  | 500 | uniform | qwen_0.5b_cachedgrad |
| Qwen2.5-0.5B-Instruct | online_hybrid | 0.6020 | 0.0429 | 0.1460 | 0.5875 | 0.5281 | 0.6709 |  |  | 1.40s |  |  |  | 500 | uniform | qwen_0.5b_online_hybrid |
