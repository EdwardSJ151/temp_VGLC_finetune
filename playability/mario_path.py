import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mario_pathing_code
import json
import os

_mario_platformer_config_cache = None

def load_mario_config(config_filename="SMB.json"):
    global _mario_platformer_config_cache
    if _mario_platformer_config_cache is not None:
        return _mario_platformer_config_cache

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_filename)
    try:
        with open(config_path) as data_file:
            _mario_platformer_config_cache = json.load(data_file)
            print(f"Mario config loaded successfully from {config_path}")
            return _mario_platformer_config_cache
    except FileNotFoundError:
        print(f"Error: Mario config file '{config_path}' not found.")
        raise
    except Exception as e:
        print(f"Error loading Mario config from '{config_path}': {e}")
        raise

def check_mario_playability(original_level_lines, platformer_config=None):
    if platformer_config is None:
        try:
            platformer_config = load_mario_config()
        except Exception:
             print("Failed to auto-load Mario config for playability check.")
             return False

    if not platformer_config:
        print("Mario platformer_config not available for playability check.")
        return False
    
    try:
        if isinstance(original_level_lines, str):
            original_level_lines = original_level_lines.split('\n')
        
        paths = mario_pathing_code.findPaths(1, platformer_config['solid'], platformer_config['jumps'], original_level_lines)
        return bool(paths)
    except Exception as e:
        print(f"Exception during Mario playability check: {e}")
        return False

if __name__ == '__main__':
    platformer_config_main = load_mario_config()

    if platformer_config_main:
        original_level_lines_main = []
        level_filename_main = "mario-1-1_unbeatable.txt"
        level_filename_without_extension_main = level_filename_main.split('.')[0]
        
        try:
            with open(level_filename_main) as level_file:
                for line in level_file:
                    original_level_lines_main.append(line.rstrip())
        except FileNotFoundError:
            print(f"Error: Level file '{level_filename_main}' not found for main execution.")
            original_level_lines_main = []

        if original_level_lines_main:
            paths_main = mario_pathing_code.findPaths(1, platformer_config_main['solid'], platformer_config_main['jumps'], original_level_lines_main)

            if paths_main:
                last_path_main = paths_main[-1]
                modified_level_chars_main = [list(line_str) for line_str in original_level_lines_main]

                for x, y in last_path_main:
                    if 0 <= y < len(modified_level_chars_main) and 0 <= x < len(modified_level_chars_main[y]):
                        modified_level_chars_main[y][x] = 'P'
                    else:
                        print(f"Warning: Path coordinate ({x},{y}) is out of level bounds.")

                final_level_output_lines_main = ["".join(line_list) for line_list in modified_level_chars_main]
                
                output_filename_main = f"{level_filename_without_extension_main}_with_path.txt"
                with open(output_filename_main, 'w') as outfile:
                    for line_str in final_level_output_lines_main:
                        outfile.write(line_str + '\n')
                print(f"Processed level saved to {output_filename_main}")
            else:
                print(f"No paths found for {level_filename_main}.")
        else:
            print(f"Level '{level_filename_main}' was empty or not found. Cannot process.")
    else:
        print("Cannot run main script logic without Mario configuration.")