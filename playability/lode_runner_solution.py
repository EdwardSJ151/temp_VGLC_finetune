from math import sqrt
from heapq import heappush, heappop
from functools import partial
import ast

def astar_shortest_path(src, isdst, adj, subOptimal, heuristic):
    dist = {}
    prev = {} 
    
    h_src = heuristic(src)
    dist[src] = h_src
    heap = [(h_src, src, h_src)]
    prev[src] = (None, None)

    pathLength = float('inf')
    final_paths_with_raw_info = []

    while heap:
        f_score_node, node_stable_state, h_score_node = heappop(heap)

        if f_score_node > pathLength + subOptimal and pathLength != float('inf'):
            continue

        if isdst(node_stable_state):
            if f_score_node < pathLength:
                pathLength = f_score_node
                path = []
                temp_s = node_stable_state
                while temp_s is not None:
                    prev_s_val, raw_leading_to_temp_s = prev[temp_s]
                    path.append((temp_s, raw_leading_to_temp_s))
                    temp_s = prev_s_val
                path.reverse()
                final_paths_with_raw_info = [path]
            elif f_score_node <= pathLength + subOptimal:
                path = []
                temp_s = node_stable_state
                while temp_s is not None:
                    prev_s_val, raw_leading_to_temp_s = prev[temp_s]
                    path.append((temp_s, raw_leading_to_temp_s))
                    temp_s = prev_s_val
                path.reverse()
                final_paths_with_raw_info.append(path)

        for move_cost, next_raw_state, next_stable_neighbor_state in adj((f_score_node, node_stable_state, h_score_node)):
            g_score_current = f_score_node - h_score_node
            g_score_neighbor = g_score_current + move_cost
            h_score_neighbor = heuristic(next_stable_neighbor_state)
            f_score_neighbor = g_score_neighbor + h_score_neighbor

            if next_stable_neighbor_state not in dist or f_score_neighbor < dist[next_stable_neighbor_state]:
                dist[next_stable_neighbor_state] = f_score_neighbor
                prev[next_stable_neighbor_state] = (node_stable_state, next_raw_state) 
                heappush(heap, (f_score_neighbor, next_stable_neighbor_state, h_score_neighbor))
    
    return final_paths_with_raw_info

_level_map_global = None
_rows_global = 0
_cols_global = 0

def set_level_data_for_helpers(level_map, rows, cols):
    global _level_map_global, _rows_global, _cols_global
    _level_map_global = level_map
    _rows_global = rows
    _cols_global = cols

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_stable_position(r_start, c_start):
    r, c = r_start, c_start
    while True:
        current_char_at_rc = _level_map_global[r][c]
        if current_char_at_rc in ('b', 'B', '#'): return (r, c)
        if r + 1 >= _rows_global: return (r, c)
        char_below = _level_map_global[r+1][c]
        if char_below in ('b', 'B', '#'): return (r, c)
        elif char_below in ('.', 'E', 'M', 'G', '-'): r += 1
        else: return (r, c)

def adj_function_for_level(node_tuple):
    _, (r, c), _ = node_tuple 
    
    adj_moves_candidates_raw = []
    ENTERABLE_TILES = ('.', '#', '-', 'G', 'M', 'E', 'b') 

    for dr, dc in [(0, -1), (0, 1), (-1, 0)]: 
        next_r, next_c = r + dr, c + dc
        if 0 <= next_r < _rows_global and 0 <= next_c < _cols_global and _level_map_global[next_r][next_c] in ENTERABLE_TILES:
            adj_moves_candidates_raw.append([1, (next_r, next_c)])

    next_r, next_c = r + 1, c
    if 0 <= next_r < _rows_global and 0 <= next_c < _cols_global and _level_map_global[next_r][next_c] in ENTERABLE_TILES:
        adj_moves_candidates_raw.append([1, (next_r, next_c)])

    if _level_map_global[r][c] == 'b':
        next_r_drop, next_c_drop = r + 1, c
        if 0 <= next_r_drop < _rows_global:
            adj_moves_candidates_raw.append([1, (next_r_drop, next_c_drop)])
            
    final_adj_list_with_raw = []
    processed_outcomes = set()

    for cost, raw_coord in adj_moves_candidates_raw:
        stable_coord = get_stable_position(raw_coord[0], raw_coord[1])
        outcome_tuple = (raw_coord, stable_coord)
        if outcome_tuple not in processed_outcomes:
            final_adj_list_with_raw.append( (cost, raw_coord, stable_coord) )
            processed_outcomes.add(outcome_tuple)
            
    return final_adj_list_with_raw

def solve_level(level_content_str):
    level_lines = level_content_str.strip().split('\n')
    parsed_level_map = [list(line) for line in level_lines]
    rows = len(parsed_level_map)
    if rows == 0: return "Error: Empty level."
    cols = len(parsed_level_map[0])
    if cols == 0: return "Error: Empty level rows."

    set_level_data_for_helpers(parsed_level_map, rows, cols)

    start_pos_initial = None
    goals_initial = []
    for r_idx, row_list in enumerate(parsed_level_map):
        for c_idx, char_val in enumerate(row_list):
            if char_val == 'M': start_pos_initial = (r_idx, c_idx)
            elif char_val == 'G': goals_initial.append((r_idx, c_idx))
    
    if not start_pos_initial: return "Error: Start position 'M' not found."
    if not goals_initial: return "Success: No goals 'G' found."

    current_player_pos_stable = get_stable_position(start_pos_initial[0], start_pos_initial[1])
    
    if _level_map_global[start_pos_initial[0]][start_pos_initial[1]] == 'M':
        _level_map_global[start_pos_initial[0]][start_pos_initial[1]] = '.'

    remaining_goals = set(goals_initial)
    full_path_taken = [current_player_pos_stable] 
    
    print(f"Starting at: {current_player_pos_stable}")
    print(f"Goals to visit: {remaining_goals}")

    while remaining_goals:
        shortest_overall_segment_astar_output = None
        goal_reached_this_iteration = None
        shortest_segment_length = float('inf')


        for target_g_pos in remaining_goals:
            is_destination_func = lambda state, tg=target_g_pos: state == tg
            heuristic_to_goal_func = lambda state, tg=target_g_pos: manhattan_distance(state, tg)
            
            paths_found_astar_output = astar_shortest_path(
                src=current_player_pos_stable,
                isdst=is_destination_func,
                adj=adj_function_for_level,
                subOptimal=0, 
                heuristic=heuristic_to_goal_func
            )

            if paths_found_astar_output:
                current_segment_astar_output = paths_found_astar_output[0]
                
                if len(current_segment_astar_output) < shortest_segment_length:
                    shortest_segment_length = len(current_segment_astar_output)
                    shortest_overall_segment_astar_output = current_segment_astar_output
                    goal_reached_this_iteration = target_g_pos
        
        if shortest_overall_segment_astar_output:
            print(f"  Path from {current_player_pos_stable} to {goal_reached_this_iteration}: {len(shortest_overall_segment_astar_output)-1} A* stable steps.")

            for k in range(1, len(shortest_overall_segment_astar_output)):
                current_stable_state_in_segment = shortest_overall_segment_astar_output[k][0]
                raw_state_initiated_from_prev = shortest_overall_segment_astar_output[k][1]

                if full_path_taken[-1] != raw_state_initiated_from_prev:
                    full_path_taken.append(raw_state_initiated_from_prev)

                r_raw, c_raw = raw_state_initiated_from_prev
                r_stable, c_stable = current_stable_state_in_segment

                if c_raw == c_stable and r_raw < r_stable:
                    for r_intermediate in range(r_raw + 1, r_stable + 1):
                        coord_to_add = (r_intermediate, c_raw)
                        if full_path_taken[-1] != coord_to_add:
                             full_path_taken.append(coord_to_add)
                elif raw_state_initiated_from_prev != current_stable_state_in_segment and \
                     full_path_taken[-1] != current_stable_state_in_segment:
                    full_path_taken.append(current_stable_state_in_segment)


            current_player_pos_stable = goal_reached_this_iteration
            remaining_goals.remove(goal_reached_this_iteration)
            
            if _level_map_global[current_player_pos_stable[0]][current_player_pos_stable[1]] == 'G':
                 _level_map_global[current_player_pos_stable[0]][current_player_pos_stable[1]] = '.'
            print(f"  Reached {goal_reached_this_iteration}. Remaining goals: {len(remaining_goals)}")
        else:
            return f"Failure: Could not find a path from {current_player_pos_stable} to any of the remaining goals: {remaining_goals}.\nPath so far: {full_path_taken}"

    if not remaining_goals:
        total_steps = len(full_path_taken) -1
        return f"Success! All goals reached. Total steps: {total_steps}.\nPath: {full_path_taken}"
    else: 
        return f"Failure: Unknown error ({remaining_goals}).\nPath so far: {full_path_taken}"

def parse_path_from_string(success_message: str) -> list | None:
    try:
        path_marker = "Path: " 
        path_so_far_marker = "Path so far: " 
        path_list_str = None
        idx_success = success_message.find(path_marker)
        if idx_success != -1: path_list_str = success_message[idx_success + len(path_marker):]
        else:
            idx_failure = success_message.find(path_so_far_marker)
            if idx_failure != -1: path_list_str = success_message[idx_failure + len(path_so_far_marker):]
        if path_list_str:
            if ']' in path_list_str: path_list_str = path_list_str[:path_list_str.rfind(']')+1]
            if path_list_str.startswith('[') and path_list_str.endswith(']'): return ast.literal_eval(path_list_str)
        return None
    except Exception: return None

def format_level_with_path_visualization(original_level_content_str: str, path_coords: list) -> str:
    level_lines = original_level_content_str.strip().split('\n')
    display_map = [list(line) for line in level_lines]
    
    rows = len(display_map)
    if rows == 0: return "Error: Empty level for visualization."
    cols = len(display_map[0])
    if cols == 0: return "Error: Empty level rows for visualization."

    for r, c in path_coords:
        if 0 <= r < rows and 0 <= c < cols:
            original_char = display_map[r][c]
            if original_char in ('.', '#', '-', 'M', 'G', 'E', 'b'):
                display_map[r][c] = 'X'
        else: 
            print(f"Warning: Path coord ({r},{c}) out of bounds for visualization.")       
            
    return "\n".join("".join(row) for row in display_map)

if __name__ == '__main__':

    # level_data = "................................\n..E.G...........................\nbBBbBBbBBBb#bbbbbbB.............\n...........#-----------.........\n...........#....bb#.............\n...........#..E.bb#......G......\n...........#....bb#...bbbbb#bbbb\n...........#....bb#........#....\n...........#....bb#........#....\n...........#....bb#.......G#....\nbbb#bbbbbbbb....bbbbbbbb#bbbbbbb\n...#....................#.......\n...#....................#.......\n...#....................#.......\nbbbbbbbbbbbbbb#bbbbbbbbb#.......\n..............#.........#.......\n..............#.........#.......\n..........E.G.#---------#..G.E..\n......#bbbbbbbbb........bbbbbbb#\n......#........................#\n......#..........M..G..........#\nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    level_data = "...MG\nBbBBB\n....G\nBBBBB"

    result_message = solve_level(level_data)
    print("\n" + "="*30)
    print(result_message)
    print("="*30)

    path_coordinates = parse_path_from_string(result_message)

    if path_coordinates:
        print("\nLevel with path visualization ('X'):")
        visualized_level = format_level_with_path_visualization(level_data, path_coordinates)
        print(visualized_level)
    elif "Success!" in result_message:
        print("\nCould not parse path from the success message for visualization.")