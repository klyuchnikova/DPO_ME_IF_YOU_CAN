# Preference comparison

| model_short | method | preference_accuracy | preference_accuracy_ci95 | delta_vs_base | mean_margin | median_margin | mean_loss | global_step | train_mean_step_sec | runtime_mean_step_sec | precompute_sec | total_train_sec | total_wall_sec | num_examples | evaluation_method | exp_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | base | 0.4560 | 0.0437 | 0.0000 | -4.8909 | -5.0309 | 0.0000 | 187 | 0.86s |  |  | 21.84m | 21.84m | 500 | raw_logprob | qwen_0.5b_base |
| Qwen2.5-0.5B-Instruct | dpo | 0.6160 | 0.0426 | 0.1600 | 1.4709 | 1.1919 | 0.6504 | 187 | 0.86s | 0.78s |  | 21.84m | 21.84m | 500 | uniform | qwen_0.5b_dpo |
| Qwen2.5-0.5B-Instruct | surprisal | 0.6300 | 0.0423 | 0.1740 | 0.9215 | 0.7195 | 0.6580 | 187 | 0.87s | 0.79s |  | 21.89m | 21.89m | 500 | uniform | qwen_0.5b_surprisal |
| Qwen2.5-0.5B-Instruct | cached_grad | 0.6260 | 0.0424 | 0.1700 | 0.6903 | 0.4342 | 0.6676 | 187 | 0.84s | 0.80s |  | 21.33m | 21.33m | 500 | uniform | qwen_0.5b_cachedgrad |
| Qwen2.5-0.5B-Instruct | online_hybrid | 0.6100 | 0.0428 | 0.1540 | 1.1922 | 0.9277 | 0.6539 | 187 | 1.25s | 0.89s |  | 31.37m | 31.37m | 500 | uniform | qwen_0.5b_online_hybrid |
