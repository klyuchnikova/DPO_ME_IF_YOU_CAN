# train dpo
python scripts/04_train.py --config configs/qwen_0.5b_dpo.yaml
# train surprisal
python scripts/04_train.py --config configs/qwen_0.5b_surprisal.yaml
# train cache
python scripts/03_precompute_cachedgrad.py --config configs/qwen_0.5b_cachedgrad.yaml --split train
python scripts/03_precompute_cachedgrad.py --config configs/qwen_0.5b_cachedgrad.yaml --split val
# train hybrid
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/04_train.py --config configs/qwen_0.5b_online_hybrid.yaml

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/03_precompute_cachedgrad.py --config configs/qwen_0.5b_cachedgrad.yaml --split test
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/04_train.py --config configs/qwen_0.5b_cachedgrad.yaml
# evaluate without internet
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/05_evaluate.py --config configs/qwen_0.5b_surprisal.yaml --checkpoint outputs/checkpoints/qwen_0.5b_surprisal/final --split test --exp-name qwen_0.5b_surprisal --num-runtime-steps 20 --num-visualize 5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/05_evaluate.py --config configs/qwen_0.5b_dpo.yaml --checkpoint outputs/checkpoints/qwen_0.5b_dpo/final --split test --exp-name qwen_0.5b_dpo --num-runtime-steps 20 --num-visualize 5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/05_evaluate.py --config configs/qwen_0.5b_cachedgrad.yaml --checkpoint outputs/checkpoints/qwen_0.5b_cachedgrad/final --split test --exp-name qwen_0.5b_cachedgrad --num-runtime-steps 20 --num-visualize 5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python scripts/05_evaluate.py \
  --config configs/qwen_0.5b_online_hybrid.yaml \
  --checkpoint outputs/checkpoints/qwen_0.5b_online_hybrid/final \
  --split test --exp-name qwen_0.5b_online_hybrid