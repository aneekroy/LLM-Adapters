#!/bin/bash

BASE=/home/models/Llama-3.2-1B-Instruct
SCRIPT=/home/aneek/LLM-Adapters/ensemble_eval_multi_lora_commonsense.py
LORAS="/home/aneek/LLM-Adapters/trained_models/llama-commonsense_170k-1B-lora,\
/home/aneek/LLM-Adapters/trained_models/llama-math_50k-1B-lora,\
/home/aneek/LLM-Adapters/trained_models/llama-alpaca_data_cleaned-1B-lora"

# Map each dataset to a specific GPU
declare -A DS_GPU=(
  [boolq]=0
  [piqa]=1
  [social_i_qa]=2
  [ARC-Challenge]=3
  [ARC-Easy]=1
  [openbookqa]=1
  [hellaswag]=1
  [winogrande]=2
)

for DS in "${!DS_GPU[@]}"; do
  GPU=${DS_GPU[$DS]}
  echo "▶ Launching $DS on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
      --dataset "$DS" \
      --model Llama-3.2-1B-Instruct \
      --base_model "$BASE" \
      --lora_weights "$LORAS" \
      --batch_size 16 \
      --ensemble_rule vote \
      --load_8bit &   # backgrounded
done

wait  # Block until all backgrounded evaluations complete
echo "✅ All evaluations finished."