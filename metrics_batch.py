import sys
import os
import json
import torch
import tqdm
import matplotlib.pyplot as plt
from utils.level_similarity_search import (
    load_json_data,
    load_level_search_index,
    search_similar_levels
)

from utils.create_img import convert_mario_to_png
from transformers import AutoProcessor, AutoModelForImageTextToText
from playability.mario_path import check_mario_playability, load_mario_config
from playability.kid_icarus_path import check_kid_icarus_playability, load_kid_icarus_config
from playability.lode_runner_solution import solve_level as lode_runner_solve_level

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_json_paths = [
    "level_generation_results_mario_vertical_20250528_140331.json",
    "level_generation_results_loderunner_horizontal_20250525_171155.json",
]

def create_level_data(level_string):
    rows = level_string.split("\n")
    return {
        "window": rows,
        "level_string": level_string
    }


def process_level(level_entry, game_type_arg, model_arg, processor_arg, index_arg, level_indices_arg, top_k_arg,
                  mario_cfg, ki_cfg):
    
    level_string = level_entry["level"]
    level_data_dict = create_level_data(level_string)

    # print(f"Processing level: {level_data_dict}")
    
    similar_indices, similarities = search_similar_levels(
        level_data_dict, model_arg, processor_arg, index_arg, level_indices_arg, game_type_arg, top_k=top_k_arg
    )
    
    level_entry["similarity_results"] = {
        "similar_levels": [int(idx) for idx in similar_indices],
        "similarity_scores": [float(score) for score in similarities]
    }

    playable_status = False
    if game_type_arg == "mario":
        if mario_cfg:
            try:
                last_line_list = list(level_data_dict["window"][-1])
                for i in range(min(4, len(last_line_list))):
                    if last_line_list[i] == '-':
                        last_line_list[i] = 'X'
                
                level_data_dict["window"][-1] = "".join(last_line_list)
                playable_status = check_mario_playability(level_data_dict["window"], mario_cfg)
            except Exception as e:
                print(f"Error during Mario playability call for level: {str(e)[:100]}...") # Log snippet
                playable_status = False 
        else:
            print("Mario config not loaded; playability check skipped.")
    elif game_type_arg in ["kid_icarus", "kid_icarus_small"]: # kid_icarus_small uses kid_icarus logic
        if ki_cfg:
            try:
                playable_status = check_kid_icarus_playability(level_data_dict["window"], ki_cfg)
            except Exception as e:
                print(f"Error during Kid Icarus playability call for level: {str(e)[:100]}...")
                playable_status = False
        else:
            print("Kid Icarus config not loaded; playability check skipped.")
    elif game_type_arg == "lode_runner":
        try:
            # lode_runner_solve_level expects the full level string
            lode_runner_result = lode_runner_solve_level(level_data_dict["level_string"])
            playable_status = lode_runner_result.startswith("Success!")
        except Exception as e:
            print(f"Error during Lode Runner playability call for level: {str(e)[:100]}...")
            playable_status = False
    elif game_type_arg == "rainbow_island":
        playable_status = True # Always playable
    else:
        print(f"Playability check not implemented for game type: {game_type_arg}")

    level_entry["playable"] = playable_status
    
    return level_entry


def get_run_metadata(run_name_str):
    run_name_lower = run_name_str.lower()
    determined_game_type = None

    if "mario" in run_name_lower:
        determined_game_type = "mario"
    elif "kid" in run_name_lower: 
        determined_game_type = "kid_icarus_small"
    elif "lode" in run_name_lower: 
        determined_game_type = "lode_runner"
    elif "rainbow" in run_name_lower: 
        determined_game_type = "rainbow_island"
    
    current_path_flag = "_path-" in run_name_str 
    return determined_game_type, current_path_flag

# game_type = "mario" # mario, kid_icarus, lode_runner, rainbow_island, kid_icarus_small
# path = True

top_k = 1

print("Loading models...")
processor_main = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
model_main = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")

mario_game_config = None
kid_icarus_game_config = None
mario_game_config = load_mario_config()
kid_icarus_game_config = load_kid_icarus_config()


file_groups = {}
print("\nPre-processing files to group them by game type and path flag...")
for input_json_path in input_json_paths:
    print(f"Reading metadata for: {input_json_path}")
    levels_data = load_json_data(input_json_path)
    
    if not levels_data:
        print(f"Warning: No data found in {input_json_path}. Skipping for grouping.")
        continue

    first_level_entry = levels_data[0]
    run_name = first_level_entry.get("run_name", "")

    if not run_name:
        print(f"Warning: 'run_name' not found in the first entry of {input_json_path}. Skipping for grouping.")
        continue

    current_game_type, current_path_flag = get_run_metadata(run_name)

    if current_game_type is None:
        print(f"Warning: Could not determine game_type from run_name: '{run_name}' in file {input_json_path}. Skipping this file.")
        continue
    
    group_key = (current_game_type, current_path_flag)
    if group_key not in file_groups:
        file_groups[group_key] = []
    file_groups[group_key].append((input_json_path, levels_data)) 
    print(f"  Assigned to group: {group_key} (Game: {current_game_type}, Path: {current_path_flag})")


for (current_game_type, current_path_flag), files_and_data_in_group in file_groups.items():
    print(f"\nProcessing group: Game Type = {current_game_type}, Path Flag = {current_path_flag}")

    if current_path_flag:
        embedding_dir = f"./embeddings/embeddings_{current_game_type}_path"
    else:
        embedding_dir = f"./embeddings/embeddings_{current_game_type}"
    
    print(f"Using embedding directory: {embedding_dir}")

    index = None
    level_indices = None
    index_file_path = os.path.join(embedding_dir, "level_index.faiss")
    index, level_indices = load_level_search_index(index_file_path)
    print(f"Successfully loaded FAISS index for {current_game_type} (Path: {current_path_flag}).")

    for input_json_path, levels_data_list in files_and_data_in_group: # levels_data is now levels_data_list
        print(f"\nProcessing file: {input_json_path} (from group {current_game_type}, Path: {current_path_flag})")
        
        print(f"Processing {len(levels_data_list)} levels...")
        processed_levels = []

        with tqdm.tqdm(total=len(levels_data_list), desc=f"Levels in {os.path.basename(input_json_path)}") as pbar:
            batch_size = 10 
            for i in range(0, len(levels_data_list), batch_size):
                batch = levels_data_list[i:i+batch_size]
                
                for level_entry in batch:
                    processed_level = process_level(
                        level_entry, current_game_type, model_main, processor_main, 
                        index, level_indices, top_k,
                        mario_game_config, kid_icarus_game_config # Pass loaded configs
                    )
                    processed_levels.append(processed_level)
                    pbar.update(1)
                
                torch.cuda.empty_cache() 

        output_json = f"with_similarity_{os.path.basename(input_json_path)}"
        with open(output_json, "w") as f:
            json.dump(processed_levels, f, indent=2)
        print(f"Results saved to {output_json}")


    print(f"Finished processing group: Game Type = {current_game_type}, Path Flag = {current_path_flag}. Releasing index.")
    del index
    del level_indices
    torch.cuda.empty_cache()

print("\nAll file groups processed successfully!")