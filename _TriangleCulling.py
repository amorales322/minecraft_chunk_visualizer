__all__ = [
    "is_block_south",
    "is_block_north",
    "is_block_east",
    "is_block_west",
    "is_block_above",
    "is_block_below",
]

import _GlobalReferences


def is_block_south(
    block_data: list[str], origin_x: int, origin_y: int, origin_z: int
) -> bool:
    if origin_z + 1 > 15:
        return False
    else:
        if (
            block_data[origin_y * 256 + (origin_z + 1) * 16 + origin_x]
            not in _GlobalReferences.blocks_ignore
        ):
            return True
    return False


def is_block_north(
    block_data: list[str], origin_x: int, origin_y: int, origin_z: int
) -> bool:
    if origin_z - 1 < 0:
        return False
    else:
        if (
            block_data[origin_y * 256 + (origin_z - 1) * 16 + origin_x]
            not in _GlobalReferences.blocks_ignore
        ):
            return True
    return False


def is_block_east(
    block_data: list[str], origin_x: int, origin_y: int, origin_z: int
) -> bool:
    if origin_x + 1 > 15:
        return False
    else:
        if (    
            block_data[origin_y * 256 + origin_z * 16 + (origin_x + 1)]
            not in _GlobalReferences.blocks_ignore
        ):
            return True
    return False


def is_block_west(
    block_data: list[str], origin_x: int, origin_y: int, origin_z: int
) -> bool:
    if origin_x - 1 < 0:
        return False
    else:
        if (
            block_data[origin_y * 256 + origin_z * 16 + (origin_x - 1)]
            not in _GlobalReferences.blocks_ignore
        ):
            return True
    return False


def is_block_above(
    block_data: list[str], origin_x: int, origin_y: int, origin_z: int
) -> bool:
    if origin_y + 1 > 15:
        return False
    else:
        if (
            block_data[(origin_y + 1) * 256 + origin_z * 16 + origin_x]
            not in _GlobalReferences.blocks_ignore
        ):
            return True
    return False


def is_block_below(
    block_data: list[str], origin_x: int, origin_y: int, origin_z: int
) -> bool:
    if origin_y - 1 < 0:
        return False
    else:
        if (
            block_data[(origin_y - 1) * 256 + origin_z * 16 + origin_x]
            not in _GlobalReferences.blocks_ignore
        ):
            return True
    return False
