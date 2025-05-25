import sys
import os
import json
import torch
import tqdm
import matplotlib.pyplot as plt
from utils.level_similarity_search import (
    load_json_data,
    load_level_search_index,
    generate_level_embedding,
    search_similar_levels
)

from utils.create_img import convert_mario_to_png
from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_level_data(level_string):
    rows = level_string.split("\n")
    processed_rows = []
    for row in rows:
        if "|" in row:
            parts = [p for p in row.split("|") if p]
            processed_rows.extend(parts)
        else:
            processed_rows.append(row)
    
    return {
        "window": rows,
        "level_string": level_string
    }


def process_level(level_entry):
    try:
        level_string = level_entry["level"]
        level_data = create_level_data(level_string)
        
        similar_indices, similarities = search_similar_levels(
            level_data, model, processor, index, level_indices, game_type, top_k=top_k
        )
        
        level_entry["similarity_results"] = {
            "similar_levels": [int(idx) for idx in similar_indices],
            "similarity_scores": [float(score) for score in similarities]
        }
        
        return level_entry
    except Exception as e:
        print(f"Error processing level: {str(e)}")
        level_entry["similarity_results"] = {
            "error": str(e)
        }
        return level_entry

input_json_paths = [
    "level_generation_results_20250522_223051.json",
]

game_type = "mario" # mario, kid_icarus, lode_runner, rainbow_island, kid_icarus_small
path = True

top_k = 5

if path:
    embedding_dir = f"../similarity_scripts/embeddings_{game_type}"
else:
    embedding_dir = f"../similarity_scripts/embeddings_{game_type}_path"

print("Loading models...")
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")

index, level_indices = load_level_search_index(f"{embedding_dir}/level_index.faiss")

for input_json_path in input_json_paths:
    print(f"\nProcessing file: {input_json_path}")
    
    levels_data = load_json_data(input_json_path)
    
    print(f"Processing {len(levels_data)} levels...")
    processed_levels = []

    with tqdm.tqdm(total=len(levels_data)) as pbar:
        batch_size = 10
        for i in range(0, len(levels_data), batch_size):
            batch = levels_data[i:i+batch_size]
            
            for level_entry in batch:
                processed_level = process_level(level_entry)
                processed_levels.append(processed_level)
                pbar.update(1)
            
            torch.cuda.empty_cache()

    output_json = f"with_similarity_{os.path.basename(input_json_path)}"
    with open(output_json, "w") as f:
        json.dump(processed_levels, f, indent=2)

    print(f"Results saved to {output_json}")

print("\nAll files processed successfully!")
