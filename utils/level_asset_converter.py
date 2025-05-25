from PIL import Image
import numpy as np
import os

# Define base tile directory
p_t = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TILE_DIR = os.path.join(p_t, "assets")

def get_chars2pngs(game_type):
    """Get character to PNG mapping based on game type"""
    
    if game_type == "mario":
        tiles_dir = os.path.join(TILE_DIR, "mario")
        chars2pngs = {
            "-": Image.open(f"{tiles_dir}/smb-background.png"),
            "X": Image.open(f"{tiles_dir}/ground.png"),
            "#": Image.open(f"{tiles_dir}/smb-unpassable.png"),
            "S": Image.open(f"{tiles_dir}/smb-breakable.png"),
            "?": Image.open(f"{tiles_dir}/special_question_block.png"),
            "Q": Image.open(f"{tiles_dir}/special_question_block.png"),
            "o": Image.open(f"{tiles_dir}/coin.png"),
            "E": Image.open(f"{tiles_dir}/blue_goomba.png"),
            "<": Image.open(f"{tiles_dir}/smb-tube-top-left.png"),
            ">": Image.open(f"{tiles_dir}/smb-tube-top-right.png"),
            "(": Image.open(f"{tiles_dir}/white_pipe_top_left.png"),
            ")": Image.open(f"{tiles_dir}/white_pipe_top_right.png"),
            "[": Image.open(f"{tiles_dir}/smb-tube-lower-left.png"),
            "]": Image.open(f"{tiles_dir}/smb-tube-lower-right.png"),
            "x": Image.open(f"{tiles_dir}/smb-path.png"),
            "Y": Image.fromarray(np.uint8(np.zeros((16, 16)))),
            "N": Image.open(f"{tiles_dir}/N.png"),
            "B": Image.open(f"{tiles_dir}/cannon_top.png"),
            "b": Image.open(f"{tiles_dir}/cannon_bottom.png"),
            "F": Image.open(f"{tiles_dir}/icon_interrogation.png"),
            "C": Image.open(f"{tiles_dir}/yellow_brick.png"),
            "U": Image.open(f"{tiles_dir}/red_brick.png"),
            "!": Image.open(f"{tiles_dir}/smb-question.png"),
            "L": Image.open(f"{tiles_dir}/life.png"),
            "2": Image.open(f"{tiles_dir}/coin2.png"),
            "g": Image.open(f"{tiles_dir}/goomba.png"),
            "G": Image.open(f"{tiles_dir}/white_goomba.png"),
            "k": Image.open(f"{tiles_dir}/koopa.png"),
            "K": Image.open(f"{tiles_dir}/winged_koopa.png"),
            "r": Image.open(f"{tiles_dir}/red_koopa.png"),
            "R": Image.open(f"{tiles_dir}/winged_red_koopa.png"),
            "y": Image.open(f"{tiles_dir}/spiny.png"),
            "t": Image.open(f"{tiles_dir}/icon_interrogation.png"),
            "T": Image.open(f"{tiles_dir}/icon_interrogation.png"),
        }
    elif game_type == "rainbow_island":
        tiles_dir = os.path.join(TILE_DIR, "rainbow_island")
        chars2pngs = {
            ".": Image.open(f"{tiles_dir}/sky2.png"),
            "B": Image.open(f"{tiles_dir}/block.png"),
            "G": Image.open(f"{tiles_dir}/ground_block.png"),
            "Y": Image.fromarray(np.uint8(np.zeros((16, 16)))),
        }
    elif game_type == "lode_runner":
        tiles_dir = os.path.join(TILE_DIR, "lode_runner")
        chars2pngs = {
            "-": Image.open(f"{tiles_dir}/rope.png"),
            "B": Image.open(f"{tiles_dir}/solid_block.png"),
            "b": Image.open(f"{tiles_dir}/breakable_block.png"),
            "#": Image.open(f"{tiles_dir}/ladder.png"),
            "G": Image.open(f"{tiles_dir}/gold.png"),
            "E": Image.open(f"{tiles_dir}/enemy.png"),
            "M": Image.open(f"{tiles_dir}/main_char.png"),
            ".": Image.fromarray(np.uint8(np.zeros((16, 16)))),
        }
    elif game_type == "kid_icarus" or game_type == "kid_icarus_small":
        tiles_dir = os.path.join(TILE_DIR, "kid_icarus")
        chars2pngs = {
            "#": Image.open(f"{tiles_dir}/block.png"),
            "D": Image.open(f"{tiles_dir}/door.png"),
            "H": Image.open(f"{tiles_dir}/lava.png"),
            "M": Image.open(f"{tiles_dir}/moving_plat.png"),
            "T": Image.open(f"{tiles_dir}/passable.png"),
            "-": Image.fromarray(np.uint8(np.zeros((16, 16)))),
        }
    else:
        raise ValueError(f"Unsupported game type: {game_type}")
    
    return chars2pngs

def create_asset_embedding(window, game_type):
    """
    Convert window of characters directly to asset-based image using the chars2pngs mapping
    Returns a PIL Image
    """
    chars2pngs = get_chars2pngs(game_type)
    
    # Get default empty space character for the game
    empty_space = "-" if game_type in ["mario", "kid_icarus"] else "."
    
    height = len(window)
    width = len(window[0]) if height > 0 else 0
    image = Image.new("RGB", (width * 16, height * 16))

    for row in range(height):
        for col in range(min(width, len(window[row]))):
            char = window[row][col]
            tile = chars2pngs.get(char, chars2pngs.get(empty_space))
            image.paste(tile, (col * 16, row * 16))

    return image