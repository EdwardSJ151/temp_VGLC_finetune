import numpy as np
import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import faiss
import json
from utils.level_asset_converter import create_asset_embedding
import tempfile
import argparse
import sys


def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_level_search_index(index_path):
    """Load the FAISS index and level indices"""
    index = faiss.read_index(index_path)
    with open(index_path + ".indices", "r") as f:
        level_indices = [int(line.strip()) for line in f]
    print(f"Index loaded from {index_path}")
    return index, level_indices


def generate_level_embedding(level_data, model, processor, game_type):
    """Process a level into an embedding"""
    window = level_data["window"]
    level_image = create_asset_embedding(window, game_type)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        level_image.save(temp_path)

    try:
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

        return embedding

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def search_similar_levels(query_level_data, model, processor, index, level_indices, game_type, top_k=5):
    """Find similar levels to the query level"""
    query_features = generate_level_embedding(query_level_data, model, processor, game_type)
    query_features = query_features.astype(np.float32).reshape(1, -1)

    faiss.normalize_L2(query_features)

    distances, indices = index.search(query_features, top_k)
    similarities = (distances + 1) / 2

    similar_level_indices = [level_indices[int(idx)] for idx in indices[0]]

    return similar_level_indices, similarities[0]


def main():
    parser = argparse.ArgumentParser(description="Search for similar levels")
    parser.add_argument("--game", type=str, required=True, choices=["mario", "kid_icarus", "lode_runner", "rainbow_island"],
                        help="Game type for similarity search")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the JSON file containing level data")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of similar levels to return")
    args = parser.parse_args()

    game_type = args.game
    data_file = args.data_file
    embedding_dir = f"embeddings_{game_type}"

    if not os.path.exists(f"{embedding_dir}/level_index.faiss"):
        print(f"Error: Embeddings for {game_type} not found in {embedding_dir}/level_index.faiss")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    index, level_indices = load_level_search_index(f"{embedding_dir}/level_index.faiss")

    print(f"Loading level data from {data_file}...")
    level_data = load_json_data(data_file)

    query_level = level_data[8]

    similar_indices, distances = search_similar_levels(
        query_level, model, processor, index, level_indices, game_type, top_k=args.top_k
    )

    print(f"Similar {game_type} levels found:")
    for idx, distance in zip(similar_indices, distances):
        print(f"Level {idx} (similarity score: {distance:.4f})")


if __name__ == "__main__":
    main()