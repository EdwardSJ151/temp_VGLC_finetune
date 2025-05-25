import math
import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image

p_t = os.path.dirname(os.path.realpath(__file__))
TILE_DIR = os.path.join(p_t, "assets")

def char_array_to_image(array, chars2pngs, target_size=None, empty_space = '-'):
    """
    Convert a 16-by-16 array of integers into a PIL.Image object
    param: array: a 16-by-16 array of integers
    """
    if target_size is None:
        image = Image.new("RGB", (array.shape[1] * 16, array.shape[0] * 16))
    else:
        image = Image.new("RGB", (target_size[1] * 16, target_size[0] * 16))
    for row in range(array.shape[0]):
        for col, char in enumerate(array[row]):
            value = chars2pngs[empty_space]
            if char in chars2pngs:
                value = chars2pngs[char]
            else:
                print(f"REPLACING {value}", (col, row))
            image.paste(value, (col * 16, row * 16))
    return image


def convert_mario_to_png(
    level: str,
    tiles_dir: str = './assets/mario',
    target_size=None,
):
    if tiles_dir is None:
        tiles_dir = TILE_DIR
    chars2pngs = {
        "-": Image.open(f"{tiles_dir}/sky2.png"),
        "X": Image.open(f"{tiles_dir}/smb-unpassable.png"),
        "S": Image.open(f"{tiles_dir}/brick2.png"),
        "?": Image.open(f"{tiles_dir}/special_question_block.png"),
        "Q": Image.open(f"{tiles_dir}/special_question_block.png"),
        "o": Image.open(f"{tiles_dir}/coin.png"),
        "E": Image.open(f"{tiles_dir}/blue_goomba.png"),
        "<": Image.open(f"{tiles_dir}/smb-tube-top-left.png"),
        ">": Image.open(f"{tiles_dir}/smb-tube-top-right.png"),
        "[": Image.open(f"{tiles_dir}/smb-tube-lower-left.png"),
        "]": Image.open(f"{tiles_dir}/smb-tube-lower-right.png"),
        "x": Image.open(f"{tiles_dir}/smb-path.png"),  # self-created
        "Y": Image.fromarray(
            np.uint8(np.zeros((16, 16)))
        ),  # black square,  # self-created
        "B": Image.open(f"{tiles_dir}/cannon_top.png"),
        "b": Image.open(f"{tiles_dir}/cannon_bottom.png"),
        "F": Image.open(f"{tiles_dir}/icon_interrogation.png"),
    }

    levels = [list(line) for line in level.split('\n')]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs, target_size, empty_space='-'), arr, level


def convert_rainbowisland_to_png(
    level: str,
    tiles_dir: str = './assets/rainbow_island',
    target_size=None,
):
    if tiles_dir is None:
        tiles_dir = TILE_DIR
    chars2pngs = {
        ".": Image.open(f"{tiles_dir}/sky2.png"),
        "B": Image.open(f"{tiles_dir}/block.png"),
        "G": Image.open(f"{tiles_dir}/ground_block.png"),
        "Y": Image.fromarray(
            np.uint8(np.zeros((16, 16)))
        ),  # black square,  # self-created
    }

    levels = [list(line) for line in level.split('\n')]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs, target_size, empty_space='.'), arr, level


def convert_loderunner_to_png(
    level: str,
    tiles_dir: str = './assets/lode_runner',
    target_size=None,
):
    if tiles_dir is None:
        tiles_dir = TILE_DIR
    chars2pngs = {
        "-": Image.open(f"{tiles_dir}/rope.png"),
        "B": Image.open(f"{tiles_dir}/solid_block.png"),
        "b": Image.open(f"{tiles_dir}/breakable_block.png"),
        "#": Image.open(f"{tiles_dir}/ladder.png"),
        "G": Image.open(f"{tiles_dir}/gold.png"),
        "E": Image.open(f"{tiles_dir}/enemy.png"),
        "M": Image.open(f"{tiles_dir}/main_char.png"),
        ".": Image.fromarray(
            np.uint8(np.zeros((16, 16)))
        ),  # black square,  # self-created
    }

    levels = [list(line) for line in level.split('\n')]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs, target_size, empty_space='.'), arr, level


def convert_kidicarus_to_png(
    level: str,
    tiles_dir: str = './assets/kid_icarus',
    target_size=None,
):
    if tiles_dir is None:
        tiles_dir = TILE_DIR
    chars2pngs = {
        "#": Image.open(f"{tiles_dir}/block.png"),
        "D": Image.open(f"{tiles_dir}/door.png"),
        "H": Image.open(f"{tiles_dir}/lava.png"),
        "M": Image.open(f"{tiles_dir}/moving_plat.png"),
        "T": Image.open(f"{tiles_dir}/passable.png"),
        "-": Image.fromarray(
            np.uint8(np.zeros((16, 16)))
        ),  # black square,  # self-created
    }

    levels = [list(line) for line in level.split('\n')]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs, target_size, empty_space='-'), arr, level