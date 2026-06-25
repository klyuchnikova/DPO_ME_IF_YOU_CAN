# Preference comparison

| method | preference_accuracy | mean_margin | median_margin | mean_loss | total_train_sec | precompute_sec | total_wall_sec | mean_step_sec | runtime_mean_step_sec | num_examples | eval_metric | exp_name | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.4560 | -4.8909 | -5.0309 | 0.0000 |  |  |  | 0.8660 |  | 500 | raw_logprob | qwen_0.5b_base | base |
| dpo | 0.6140 | 1.1818 | 0.9239 | 0.6628 |  |  |  |  | 0.8338 | 500 |  | qwen_0.5b_dpo | outputs/checkpoints/qwen_0.5b_dpo/final |
| surprisal | 0.5980 | 0.6116 | 0.4880 | 0.6625 |  |  |  |  | 0.8013 | 500 |  | qwen_0.5b_surprisal | outputs/checkpoints/qwen_0.5b_surprisal/final |
| cached_grad | 0.6020 | 0.6679 | 0.5589 | 0.6719 |  |  |  |  | 0.9274 | 500 | uniform | qwen_0.5b_cachedgrad | outputs/checkpoints/qwen_0.5b_cachedgrad/final |
| online_hybrid | 0.6020 | 0.5875 | 0.5281 | 0.6709 |  |  |  |  | 1.4036 | 500 | uniform | qwen_0.5b_online_hybrid | outputs/checkpoints/qwen_0.5b_online_hybrid/final |
