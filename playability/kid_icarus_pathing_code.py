import pathfinding

def _makeIsStandable(all_standable_tiles_from_config):
    def isStandable(tile_char):
        return tile_char in all_standable_tiles_from_config
    return isStandable

def _makeIsTrulySolid(all_standable_tiles_from_config, passable_platform_tiles_from_config):
    truly_solid_tiles = [
        t for t in all_standable_tiles_from_config if t not in passable_platform_tiles_from_config
    ]
    def isTrulySolid(tile_char):
        return tile_char in truly_solid_tiles
    return isTrulySolid

def makeGetNeighbors(jumps_config, levelStr, visited_set_ref,
                     isStandableFunc, isTrulySolidFunc):
    maxX = len(levelStr[0]) - 1
    maxY = len(levelStr) - 1
    
    jumpDiffs = []
    for jump_arc in jumps_config:
        jumpDiff = [jump_arc[0]]
        for ii in range(1, len(jump_arc)):
            jumpDiff.append((jump_arc[ii][0] - jump_arc[ii-1][0], jump_arc[ii][1] - jump_arc[ii-1][1]))
        jumpDiffs.append(jumpDiff)
    processed_jumps = jumpDiffs

    def wrap_x(x_coord):
        return x_coord % (maxX + 1)

    def getNeighbors(astar_node_wrapper):
        current_g_score = astar_node_wrapper[0]
        actual_pos = astar_node_wrapper[1] 
        
        x, y, jump_idx, jump_step, jump_dir = actual_pos
        
        visited_set_ref.add(actual_pos) 

        neighbors = []
        
        if jump_idx != -1:
            next_jump_step = jump_step + 1
            current_jump_arc = processed_jumps[jump_idx]

            if next_jump_step < len(current_jump_arc):
                delta_x_step = current_jump_arc[next_jump_step][0]
                delta_y_step = current_jump_arc[next_jump_step][1]
                
                target_x_raw = x + (jump_dir * delta_x_step)
                target_y = y + delta_y_step
                target_x_wrapped = wrap_x(target_x_raw)

                if target_y < 0: 
                    if not isTrulySolidFunc(levelStr[0][target_x_wrapped]):
                        neighbors.append([current_g_score + 1, (target_x_wrapped, 0, jump_idx, next_jump_step, jump_dir)])
                elif target_y <= maxY: 
                    if not isTrulySolidFunc(levelStr[target_y][target_x_wrapped]):
                        neighbors.append([current_g_score + 1, (target_x_wrapped, target_y, jump_idx, next_jump_step, jump_dir)])
        
        on_ground = False
        if y + 1 <= maxY:
            if isStandableFunc(levelStr[y + 1][x]):
                on_ground = True
        
        if on_ground:
            for dx_walk in [-1, 1]:
                walk_target_x_raw = x + dx_walk
                walk_target_x_wrapped = wrap_x(walk_target_x_raw)
                if not isTrulySolidFunc(levelStr[y][walk_target_x_wrapped]):
                    neighbors.append([current_g_score + 1, (walk_target_x_wrapped, y, -1, 0, 0)])

            for new_jump_idx in range(len(processed_jumps)):
                jump_first_step_dx = processed_jumps[new_jump_idx][0][0]
                jump_first_step_dy = processed_jumps[new_jump_idx][0][1]

                for new_jump_dir in [-1, 1]:
                    
                    jump_target_x_raw = x + (new_jump_dir * jump_first_step_dx)
                    jump_target_y = y + jump_first_step_dy
                    jump_target_x_wrapped = wrap_x(jump_target_x_raw)

                    if jump_target_y < 0:
                        if not isTrulySolidFunc(levelStr[0][jump_target_x_wrapped]):
                            neighbors.append([current_g_score + 1, (jump_target_x_wrapped, 0, new_jump_idx, 0, new_jump_dir)])
                    elif jump_target_y <= maxY:
                        if not isTrulySolidFunc(levelStr[jump_target_y][jump_target_x_wrapped]):
                            neighbors.append([current_g_score + 1, (jump_target_x_wrapped, jump_target_y, new_jump_idx, 0, new_jump_dir)])
        else:
            fall_target_y = y + 1
            if fall_target_y <= maxY: 
                if not isTrulySolidFunc(levelStr[fall_target_y][x]):
                    neighbors.append([current_g_score + 1, (x, fall_target_y, -1, 0, 0)])
                
                for dx_fall in [-1, 1]:
                    fall_diag_target_x_raw = x + dx_fall
                    fall_diag_target_x_wrapped = wrap_x(fall_diag_target_x_raw)
                    if not isTrulySolidFunc(levelStr[fall_target_y][fall_diag_target_x_wrapped]):
                        neighbors.append([current_g_score + 1.4, (fall_diag_target_x_wrapped, fall_target_y, -1, 0, 0)])
        return neighbors
    return getNeighbors

def findPaths(subOptimal, config_solids, config_passable, config_jumps, levelStr):
    if not levelStr or not levelStr[0]:
        print("Error: Level string is empty.")
        return []

    isStandable = _makeIsStandable(config_solids)
    isTrulySolid = _makeIsTrulySolid(config_solids, config_passable)
    
    maxX = len(levelStr[0]) - 1
    maxY = len(levelStr) - 1

    start_node_details = None
    for r in range(maxY -1, -1, -1):
        for c in range(maxX + 1):
            if levelStr[r][c] == '-' and isStandable(levelStr[r+1][c]):
                start_node_details = (c, r, -1, 0, 0)
                break
        if start_node_details:
            break

    if not start_node_details:
        for c in range(maxX + 1):
            if levelStr[maxY][c] == '-':
                start_node_details = (c, maxY, -1, 0, 0)
                break
    
    if not start_node_details:
        print("Error: No valid starting empty space ('-') found according to the new criteria.")
        return []

    visited_in_exploration = set()
    getNeighbors_exploration = makeGetNeighbors(config_jumps, levelStr, visited_in_exploration, isStandable, isTrulySolid)
    
    optimistic_goal_condition = lambda actual_pos: actual_pos[1] == 0 and \
                                               isStandable(levelStr[actual_pos[1]][actual_pos[0]])
    
    exploration_heuristic = lambda actual_pos: 0

    pathfinding.astar_shortest_path(
        start_node_details,
        optimistic_goal_condition,
        getNeighbors_exploration,
        max(10, subOptimal),
        exploration_heuristic
    )

    if not visited_in_exploration:
        print("Warning: Exploration phase visited no states. Start node might be invalid or trapped immediately.")
        return []

    highest_y_reached = maxY + 1
    actual_target_node_details = None

    for visited_node_details in visited_in_exploration:
        vx, vy, _, _, _ = visited_node_details
        if isStandable(levelStr[vy][vx]):
            if vy < highest_y_reached:
                highest_y_reached = vy
                actual_target_node_details = visited_node_details
            elif vy == highest_y_reached and actual_target_node_details and vx < actual_target_node_details[0]:
                 actual_target_node_details = visited_node_details


    if not actual_target_node_details:
        print("No standable position was reached during exploration.")
        return []
    
    visited_for_reconstruction = set()
    getNeighbors_reconstruction = makeGetNeighbors(config_jumps, levelStr, visited_for_reconstruction, isStandable, isTrulySolid)
    
    precise_goal_condition = lambda current_pos_details: current_pos_details == actual_target_node_details
    
    reconstruction_heuristic = lambda current_pos_details: \
        abs(current_pos_details[0] - actual_target_node_details[0]) + \
        abs(current_pos_details[1] - actual_target_node_details[1])

    paths_to_highest_found = pathfinding.astar_shortest_path(
        start_node_details,
        precise_goal_condition,
        getNeighbors_reconstruction,
        subOptimal,
        reconstruction_heuristic
    )
    
    if not paths_to_highest_found:
        print(f"Exploration found highest point {actual_target_node_details}, but failed to reconstruct a path to it. This is unexpected.")
        return []
        
    processed_paths = []
    for path_wrapper_nodes in paths_to_highest_found:
        if path_wrapper_nodes:
            coord_path = [(node_wrapper[0], node_wrapper[1]) for node_wrapper in path_wrapper_nodes]
            processed_paths.append(coord_path)
    return processed_paths


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <kid_icarus_config.json> <level.txt>")
        sys.exit(1)

    config_filename = sys.argv[1]
    level_filename = sys.argv[2]

    try:
        with open(config_filename) as f:
            ki_config = json.load(f)
    except Exception as e:
        print(f"Error loading JSON config {config_filename}: {e}")
        sys.exit(1)

    level_data = []
    try:
        with open(level_filename) as f:
            for line in f:
                level_data.append(line.rstrip())
        if not level_data or not level_data[0]:
            print(f"Error: Level file {level_filename} is empty or malformed.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading level file {level_filename}: {e}")
        sys.exit(1)

    missing_keys = [key for key in ['solid', 'jumps', 'passable'] if key not in ki_config]
    if missing_keys:
        print(f"Error: JSON config missing keys: {', '.join(missing_keys)}.")
        sys.exit(1)

    print(f"Attempting to find highest path in {level_filename} using {config_filename}...")
    found_paths = findPaths(1, ki_config['solid'], ki_config['passable'], ki_config['jumps'], level_data)

    if found_paths:
        print(f"Found {len(found_paths)} path(s) to the highest reachable point.")
        
        last_path_coords = found_paths[-1]
        
        if not level_data:
             print("Error: level_data is empty, cannot modify.")
             sys.exit(1)
        modified_level_chars = [list(str(line_str)) for line_str in level_data]


        for x_coord, y_coord in last_path_coords:
            if 0 <= y_coord < len(modified_level_chars) and \
               0 <= x_coord < len(modified_level_chars[y_coord]):
                modified_level_chars[y_coord][x_coord] = 'P'
        
        output_level_str = ["".join(line_list) for line_list in modified_level_chars]
        
        base_name = level_filename.split('.')[0]
        if '/' in base_name:
            base_name = base_name.split('/')[-1]
        output_filename = f"{base_name}_with_KI_highest_path.txt"
        
        output_filepath = output_filename


        with open(output_filepath, 'w') as outfile:
            for line_str in output_level_str:
                outfile.write(line_str + '\n')
        print(f"Kid Icarus highest path saved to {output_filepath}")
        print(f"Path taken (last one): {last_path_coords}")
        if last_path_coords:
             print(f"Highest point reached: y={last_path_coords[-1][1]}")


    else:
        print(f"No paths found for Kid Icarus in {level_filename} to any reachable high point.")
