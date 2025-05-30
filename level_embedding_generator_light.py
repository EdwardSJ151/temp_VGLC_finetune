import numpy as np
import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import faiss
from tqdm import tqdm
import os
from level_asset_converter import create_asset_embedding
import argparse
import tempfile
import sys


def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_faiss_index(dimension, output_path):
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    return index


def save_level_indices(level_indices, output_path):
    with open(output_path + ".indices", "a") as f:
        for idx in level_indices:
            f.write(f"{idx}\n")

def process_level(level, model, processor, game_type):
    window = level["window"]
    level_image = create_asset_embedding(window, game_type)

    # level_image.save(f"debug_{game_type}_level.png")
    # raise Exception("olha imagem.")

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        level_image.save(temp_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": f"{temp_path}"},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.image_hidden_states.cpu().numpy()
        embedding = embedding.reshape(-1)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return embedding


def main():
    parser = argparse.ArgumentParser(description="Generate level embeddings")
    parser.add_argument("--game", type=str, required=True, choices=["mario", "kid_icarus", "lode_runner", "rainbow_island"],
                        help="Game type for embedding generation")
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to the JSON file containing level data")
    parser.add_argument("--path", action="store_true", help="When set, appends '_path' to the embedding directory name")
    args = parser.parse_args()
    
    game_type = args.game
    data_file = args.data_file
    
    embedding_dir = os.path.join("embeddings", f"embeddings_{game_type}")
    if args.path:
        embedding_dir += "_path"
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model.to("cuda")

    print(f"Loading level data from {data_file} for {game_type}...")
    # Load just level 0 to get dimensions
    with open(data_file, "r", encoding="utf-8") as f:
        data_json = json.load(f)
        first_level = data_json[0]
        total_levels = len(data_json)
        del data_json  # Free memory

    os.makedirs(embedding_dir, exist_ok=True)

    first_embedding = process_level(first_level, model, processor, game_type)
    dimension = len(first_embedding)

    print("Creating FAISS index...")
    index = create_faiss_index(dimension, f"{embedding_dir}/level_index.faiss")

    print("Processing levels one by one...")
    
    for level_idx in tqdm(range(total_levels)):
        with open(data_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            level_data = all_data[level_idx]
            del all_data  # Free memory
        
        # Process level
        embedding = process_level(level_data, model, processor, game_type)
        
        # Convert to float32 for FAISS
        embedding_array = np.array([embedding]).astype(np.float32)
        faiss.normalize_L2(embedding_array)
        
        index.add_with_ids(embedding_array, np.array([level_idx]))
        
        save_level_indices([level_idx], f"{embedding_dir}/level_index.faiss")
        
        if level_idx % 100 == 0 or level_idx == total_levels - 1:
            faiss.write_index(index, f"{embedding_dir}/level_index.faiss")
        
        del embedding
        del embedding_array
        torch.cuda.empty_cache()
        
    print(f"Done! Embeddings and index saved to {embedding_dir}/")


if __name__ == "__main__":
    main()