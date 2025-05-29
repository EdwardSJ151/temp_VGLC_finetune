#!/bin/bash

RESULTS_DIR="inference_results"
PYTHON_SCRIPT="metrics_batch.py"

declare -a mario_files
declare -a kid_icarus_files
declare -a lode_runner_files
declare -a rainbow_island_files

if [ ! -d "$RESULTS_DIR" ]; then
    exit 1
fi

if ! command -v python &> /dev/null; then
    exit 1
fi

for file_path in "$RESULTS_DIR"/*.json; do
    [ -e "$file_path" ] || { exit 1; }

    filename=$(basename "$file_path")
    filename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')

    if [[ "$filename_lower" == *"mario"* ]]; then
        mario_files+=("$file_path")
    elif [[ "$filename_lower" == *"kid"* ]]; then
        kid_icarus_files+=("$file_path")
    elif [[ "$filename_lower" == *"lode"* ]]; then
        lode_runner_files+=("$file_path")
    elif [[ "$filename_lower" == *"rainbow"* ]]; then
        rainbow_island_files+=("$file_path")
    fi
done

if [ ${#mario_files[@]} -gt 0 ]; then
    python "$PYTHON_SCRIPT" --input_json_paths "${mario_files[@]}" &
fi

if [ ${#kid_icarus_files[@]} -gt 0 ]; then
    python "$PYTHON_SCRIPT" --input_json_paths "${kid_icarus_files[@]}" &
fi

if [ ${#lode_runner_files[@]} -gt 0 ]; then
    python "$PYTHON_SCRIPT" --input_json_paths "${lode_runner_files[@]}" &
fi

if [ ${#rainbow_island_files[@]} -gt 0 ]; then
    python "$PYTHON_SCRIPT" --input_json_paths "${rainbow_island_files[@]}" &
fi

echo "Todos os processos foram iniciados"
wait
echo "Concluido"