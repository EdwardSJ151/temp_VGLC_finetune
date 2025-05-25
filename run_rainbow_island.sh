#!/bin/bash
CONFIG_PATH1="./rainbow_island/grid_search.yaml"
CONFIG_PATH2="./rainbow_island/grid_search_gemma.yaml"
echo "Running finetune with ${CONFIG_PATH1}"
python unsloth_finetune_and_inference.py --config_path "${CONFIG_PATH1}"
echo "Running finetune with ${CONFIG_PATH2}"
python unsloth_finetune_and_inference.py --config_path "${CONFIG_PATH2}"
exit $?