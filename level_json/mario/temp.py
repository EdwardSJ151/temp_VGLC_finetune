import random
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats


STATISTICS = {
    "enemy": np.array([1.0, 3.0, 7.0]),
    "pipe": np.array([0.0, 2.0, 5.0]),
    "block": np.array([50.0, 75.0, 176.0]),
}

PREFIX_LIST = [
    'Generate a level with', 
    'Create a level that has', 
    'Please generate a level with', 
    'Build a level featuring', 
    'Create a level featuring', 
    'Give me a level that includes', 
    'Make a level that includes', 
    'Produce a level featuring', 
    'Craft a level that includes', 
    'I need a level that has', 
    'I want to see a level that includes', 
    'Design a level with', 
    'I want a level with', 
    'Make a level with', 
    'Design a level around', 
    'Generate levels with', 
    'Construct a level featuring', 
    'I want a level that has', 
    'I would like a level that has'
]


class Prompter:
    def __init__(
        self,
        use_raw_counts: bool = True,
        statistics: Optional[Dict[str, Any]] = None,
        prefix_list: Optional[List[str]] = None,
    ):

        self.use_raw_counts = use_raw_counts
        self.statistics = statistics
        if statistics is None:
            self.statistics = STATISTICS
        
        self.prefix_list = prefix_list
        if prefix_list is None:
            self.prefix_list = PREFIX_LIST

    @property
    def pipe_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["pipe"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def enemy_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["enemy"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords

    @property
    def block_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["block"]
        keywords = ["little", "little", "some", "many"]
        return thresholds, keywords

    def count_pipes(self, flattened_level: str) -> int:
        return flattened_level.count("<>")

    def count_enemies(self, flattened_level: str) -> int:
        return flattened_level.count("E") + flattened_level.count("B")

    def count_blocks(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["X", "S", "?", "Q"]])

    def _flatten_level(self, string_level: str) -> str:
        return string_level

    def pipe_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_pipes(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.pipe_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} pipes", keyword

    def enemy_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_enemies(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.enemy_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} enemies", keyword

    def block_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_blocks(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} blocks", keyword

    def elevation_prompt(self, newline_level: str, level: str):
        rows = newline_level.split('\n')
        top_rows = rows[:6]
        for row in top_rows:
            if "X" in row or "<" in row or ">" in row:
                return "high elevation", "high"
        return "low elevation", "low"

    def dataset_statistics(self, dataset):
        pass
    
    def get_random_prefix(self):
        """Get a random prefix from the prefix list."""
        return random.choice(self.prefix_list)

    def __call__(
        self, level_json: Dict = None, sample_prompt: bool = False
    ) -> Union[str, Dict]:
        flat_level = level_json["str_horizontal_nosplit"]
        newline_level = level_json["str_horizontal_newline"]
        
        pipe_prompt, pipe_keyword = self.pipe_prompt(flat_level, flat_level)
        enemy_prompt, enemy_keyword = self.enemy_prompt(flat_level, flat_level)
        block_prompt, block_keyword = self.block_prompt(flat_level, flat_level)
        elevation_prompt, elevation_keyword = self.elevation_prompt(newline_level, flat_level)

        prompt = f"{pipe_prompt}, {enemy_prompt}, {block_prompt}, {elevation_prompt}"
        
        prefix = self.get_random_prefix()
        full_prompt = f"{prefix} {prompt}"

        prompt_dict = {
            "pipe": pipe_prompt,
            "enemy": enemy_prompt,
            "block": block_prompt,
            "elevation": elevation_prompt,
            "full_prompt": full_prompt
        }
        
        return prompt, prompt_dict, flat_level

    def process_json_file(self, json_file_path: str, output_file_path: str = None):
        """
        Process all levels in a JSON file and add prompt information to each level.
        
        Args:
            json_file_path: Path to the input JSON file
            output_file_path: Path to save the updated JSON (if None, updates original)
        """
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        for level in data:
            prompt, prompt_dict, _ = self(level)
            level["prompt"] = prompt
            level["prompt_dict"] = prompt_dict
            level["full_prompt"] = prompt_dict["full_prompt"]
        
        if output_file_path is None:
            output_file_path = json_file_path
        
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Mario levels JSON file")
    parser.add_argument("--input", type=str, help="Input JSON file path", required=True)
    parser.add_argument("--output", type=str, help="Output JSON file path (optional)")
    parser.add_argument("--use_raw", action="store_true", help="Use raw counts instead of keywords")
    parser.add_argument("--seed", type=int, help="Random seed for consistent prefix selection", default=None)
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    prompter = Prompter(use_raw_counts=args.use_raw)
    updated_data = prompter.process_json_file(args.input, args.output)
    
    print(f"Processed {len(updated_data)} levels")
    print(f"Sample prompt for first level: {updated_data[0]['prompt']}")
    print(f"Sample full prompt for first level: {updated_data[0]['full_prompt']}")