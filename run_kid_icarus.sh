#!/bin/bash

PYTHON_SCRIPT="unsloth_finetune_and_inference.py"

CONFIG_FILES_KID_ICARUS=(
    "kid_icarus/grid_search.qwen2.5.yaml"
    "kid_icarus/grid_search_qwen3.yaml"
    "kid_icarus/grid_search_llama.yaml"
)

ALL_CONFIG_FILES=("${CONFIG_FILES_KID_ICARUS[@]}")

for config_file in "${ALL_CONFIG_FILES[@]}"; do
    echo "Starting run with config: $config_file"
    python "$PYTHON_SCRIPT" --config_path "$config_file" &
done

echo "Rodando"
wait
echo "Kid Icarus concluido"
