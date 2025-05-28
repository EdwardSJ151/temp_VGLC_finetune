#!/bin/bash

PYTHON_SCRIPT="unsloth_finetune_and_inference.py"

CONFIG_FILES_LODE_RUNNER=(
    "lode_runner/grid_search.qwen2.5.yaml"
    "lode_runner/grid_search_qwen3.yaml"
    "lode_runner/grid_search_llama.yaml"
    "lode_runner/grid_search_gemma.yaml"
)

ALL_CONFIG_FILES=("${CONFIG_FILES_LODE_RUNNER[@]}")

for config_file in "${ALL_CONFIG_FILES[@]}"; do
    echo "Starting run with config: $config_file"
    python "$PYTHON_SCRIPT" --config_path "$config_file" &
done

echo "Rodando"
wait
echo "Lode Runner concluido"
