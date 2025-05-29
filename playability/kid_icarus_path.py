import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kid_icarus_pathing_code # This script needs to be in the same directory or Python path
import json
import collections # Already imported in the original

# --- Start of functions for metrics_batch.py ---
_kid_icarus_platformer_config_cache = None

def load_kid_icarus_config(config_filename="KI.json"):
    """Loads the Kid Icarus platformer configuration from a JSON file."""
    global _kid_icarus_platformer_config_cache
    if _kid_icarus_platformer_config_cache is not None:
        return _kid_icarus_platformer_config_cache

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_filename)
    try:
        with open(config_path) as data_file:
            _kid_icarus_platformer_config_cache = json.load(data_file)
            print(f"Kid Icarus config loaded successfully from {config_path}")
            return _kid_icarus_platformer_config_cache
    except FileNotFoundError:
        print(f"Error: Kid Icarus config file '{config_path}' not found.")
        raise
    except Exception as e:
        print(f"Error loading Kid Icarus config from '{config_path}': {e}")
        raise

def check_kid_icarus_playability(original_level_lines, platformer_config=None):
    if platformer_config is None:
        platformer_config = load_kid_icarus_config()
        
    try:
        if isinstance(original_level_lines, str):
            original_level_lines = original_level_lines.split('\n')

        paths = kid_icarus_pathing_code.findPaths(
            1,
            platformer_config['solid'],
            platformer_config['passable'],
            platformer_config['jumps'],
            original_level_lines
        )
        if not paths:
            return False

        modified_level_chars = [list(line_str) for line_str in original_level_lines]
        last_path = paths[-1]
        for x, y in last_path:
            if 0 <= y < len(modified_level_chars) and 0 <= x < len(modified_level_chars[y]):
                modified_level_chars[y][x] = 'P'


        path_marked_level_lines = ["".join(line_list) for line_list in modified_level_chars]
        

        filled_level_lines = fill_inaccessible_areas(
            original_grid_map=path_marked_level_lines,
            start_char='P',
            solid_chars=platformer_config.get('solid', ['#', 'H']),
            fill_char='#'
        )
        
        standable_chars_for_check = platformer_config.get('standable_for_check', ['#', 'T', 'M'])
        is_valid_according_to_logic = can_find_standable_platform_above_topmost_start(
            grid_map=filled_level_lines,
            start_char='P',
            standable_chars=standable_chars_for_check
        )
        
        return not is_valid_according_to_logic # De proposito, se tem plataforma acima, o level nao e valido

    except Exception as e:
        print(f"Exception during Kid Icarus detailed playability check: {e}")
        return False


def get_all_reachable_cells(grid_map, start_char='M', solid_chars=['#', 'H'], allow_wrap=True):
    if not grid_map or not grid_map[0]:
        return set()

    if solid_chars is None:
        current_solid_chars_set = {'#'}
    else:
        current_solid_chars_set = set(solid_chars)

    rows = len(grid_map)
    cols = len(grid_map[0])

    start_nodes = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid_map[r_idx][c_idx] == start_char:
                start_nodes.append((r_idx, c_idx))

    if not start_nodes:
        return set()

    queue = collections.deque()
    reachable_coords = set() 

    for r_start, c_start in start_nodes:
        if (r_start, c_start) not in reachable_coords:
            queue.append((r_start, c_start))
            reachable_coords.add((r_start, c_start))
    
    while queue:
        r, c = queue.popleft()
        
        next_candidate_cells = []
        if r > 0: next_candidate_cells.append((r - 1, c))
        if r < rows - 1: next_candidate_cells.append((r + 1, c))

        can_wrap_this_row = False
        if allow_wrap and cols > 0: 
            if grid_map[r][0] not in current_solid_chars_set and \
               grid_map[r][cols - 1] not in current_solid_chars_set:
                can_wrap_this_row = True
        
        if can_wrap_this_row and c == 0: 
            next_candidate_cells.append((r, cols - 1)) 
        elif c > 0: 
            next_candidate_cells.append((r, c - 1))

        if can_wrap_this_row and c == cols - 1: 
            next_candidate_cells.append((r, 0)) 
        elif c < cols - 1: 
            next_candidate_cells.append((r, c + 1))

        for nr, nc in next_candidate_cells:
            if (nr, nc) not in reachable_coords and \
               grid_map[nr][nc] not in current_solid_chars_set:
                reachable_coords.add((nr, nc))
                queue.append((nr, nc))
                
    return reachable_coords

def fill_inaccessible_areas(
    original_grid_map,
    start_char='M',
    solid_chars=['#', 'H'], 
    fill_char='#',    
    allow_wrap=True
):
    if not original_grid_map or not original_grid_map[0]:
        return [] 

    rows = len(original_grid_map)
    cols = len(original_grid_map[0])

    if solid_chars is None:
        current_solid_chars_set = {'#'}
    else:
        current_solid_chars_set = set(solid_chars)

    reachable_coords = get_all_reachable_cells(
        grid_map=original_grid_map,
        start_char=start_char,
        solid_chars=list(current_solid_chars_set),
        allow_wrap=allow_wrap
    )

    filled_map_lines = []
    for r_idx in range(rows):
        original_row_str = original_grid_map[r_idx]
        new_row_char_list = list(original_row_str)

        for c_idx in range(cols):
            current_char_at_cell = original_row_str[c_idx]
            coord = (r_idx, c_idx)

            if coord in reachable_coords:
                pass
            else:
                if current_char_at_cell in current_solid_chars_set:
                    pass 
                else:

                    new_row_char_list[c_idx] = fill_char
        
        filled_map_lines.append("".join(new_row_char_list))

    return filled_map_lines


def can_find_standable_platform_above_topmost_start(
    grid_map,
    start_char='P',
    standable_chars=['#', 'T', "M"]
):

    if not grid_map or not grid_map[0]:
        return False

    if standable_chars is None:
        current_standable_set = {'#'}
    else:
        current_standable_set = set(standable_chars)

    rows = len(grid_map)
    cols = len(grid_map[0])
    topmost_start_char_row = -1

    for r_idx in range(rows):
        if start_char in grid_map[r_idx]:
            topmost_start_char_row = r_idx
            break

    if topmost_start_char_row == -1:
        return False 
    if topmost_start_char_row == 0: # Cannot be above if player is on the first row
        return False

    # Iterate rows *above* the topmost_start_char_row up to the top of the map
    # Player pos is r_player_pos, ground_pos is r_player_pos + 1
    # So, if player is at r_player_pos, ground is r_player_pos+1.
    # We are looking for a situation where a player *could* be (empty space)
    # above a standable platform.
    for r_player_pos in range(topmost_start_char_row): # Check rows 0 to topmost_start_char_row - 1
        player_row_str = grid_map[r_player_pos]
        # Ground for this potential player position is the row below it
        ground_row_str = grid_map[r_player_pos + 1] 
        
        for c_idx in range(cols):
            char_in_player_cell = player_row_str[c_idx]
            char_in_ground_cell = ground_row_str[c_idx]
            # If the cell where player would be is NOT standable (e.g. empty)
            # AND the cell below it IS standable
            if char_in_player_cell not in current_standable_set and \
               char_in_ground_cell in current_standable_set:
                return True # Found such a configuration

    return False


if __name__ == '__main__':
    platformer_config_main = load_kid_icarus_config() 

    if platformer_config_main:
        original_level_lines_main = []
        
        level_string_main = "------\n-----------\n-----------" 
        original_level_lines_main = level_string_main.split('\n')
        level_filename_without_extension_main = "string_level"

        # level_filename_main = "kidicarus_3.txt" 
        # level_filename_without_extension_main = level_filename_main.split('.')[0]
        #   with open(level_filename_main) as level_file:
        #       for line in level_file:
        #           original_level_lines_main.append(line.rstrip())



        if not original_level_lines_main:
            print(f"Error: Level data is empty. Cannot process.")
        else:
            paths_main = kid_icarus_pathing_code.findPaths(
                1,
                platformer_config_main['solid'],
                platformer_config_main['passable'],
                platformer_config_main['jumps'],
                original_level_lines_main
            )

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

                final_level_filled_main = fill_inaccessible_areas(
                    original_grid_map=final_level_output_lines_main,
                    fill_char="#"
                )

                result_bool_main = can_find_standable_platform_above_topmost_start(final_level_filled_main)
                print(f"Can find standable platform above topmost path ('P') start: {result_bool_main}")
                
                with open(output_filename_main, 'w') as outfile:
                    for line_str in final_level_filled_main:
                        outfile.write(line_str + '\n')
                print(f"Processed level (filled, with path) saved to {output_filename_main}")
            else:
                print(f"No paths found for the level using Kid Icarus logic.")
    else:
        print("Cannot run main script logic without Kid Icarus configuration.")