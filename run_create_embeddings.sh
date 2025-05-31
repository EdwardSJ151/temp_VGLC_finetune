#!/bin/bash

PYTHON_SCRIPT="level_embedding_generator_light.py"

declare -a RUNS=(
    "--game mario --data_file level_json/mario/mario1and2_indexed.json"
    "--game mario --data_file level_json/mario/mario1and2_paths_indexed.json --path"
    "--game kid_icarus_small --data_file level_json/kid_icarus/kid_icarus_small_indexed.json"
    "--game kid_icarus_small --data_file level_json/kid_icarus/kid_icarus_small_paths_indexed.json --path"
    "--game lode_runner --data_file level_json/lode_runner/lode_runner_indexed.json"
    "--game rainbow_island --data_file level_json/rainbow_island/rainbow_island_indexed.json"
)

for run_args in "${RUNS[@]}"; do
    echo "Starting process: $PYTHON_SCRIPT $run_args"
    python "$PYTHON_SCRIPT" $run_args &
done

echo "Running all embedding processes..."
wait
echo "All embedding processes completed."