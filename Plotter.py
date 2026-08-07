__all__ = ["plot", "_generate_subchunk_mesh"]

import pyvista as pv
import numpy as np
from itertools import product as iter_product

import program_references


def plot(
    mesh_data: list[pv.PolyData], wireframe: bool = False, unit_test_mode: bool = False
) -> None:
    """
    Plots the mesh data using the pyvista Plotter class.

    :param mesh_data: List of pyvista PolyData objects to plot
    :type mesh_data: list[pyvista.PolyData]
    :param wireframe: Enable/Disable wireframe.
    :type wireframe: bool
    :param unit_test_mode: Used for unit testing. Verifies the function works and skips plotting.
    :type unit_test_mode: bool
    :return: `None`
    :rtype: None
    """
    pl = pv.Plotter()
    if len(mesh_data) != 0:
        for mesh in mesh_data:
            pl.add_mesh(mesh, show_edges=wireframe)
    else:
        print("Mesh is empty")
    if not unit_test_mode:
        pl.camera_position = "zx"
        pl.window_size = [1920, 1080]
        pl.add_axes(
            interactive=True,
            line_width=5,
        )
        pl.show()


def _get_block(
    palette_data: dict[str, str], block_data: list[int], x: int, y: int, z: int
) -> bool:
    if (
        15 >= z >= 0
        and 15 >= y >= 0
        and 15 >= x >= 0
        and palette_data[block_data[y * 256 + z * 16 + x]]["Name"]
        not in program_references.blocks_ignore
    ):
        return True
    return False


def _generate_point_data(
    mult_factor: int,
    origin: tuple[int, int, int],
    offset: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    return np.array(
        (
            (
                0.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                0.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                0.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                1.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
            (
                1.0 * mult_factor + origin[0] + offset[0],
                0.0 * mult_factor + origin[1] + offset[1],
                1.0 * mult_factor + origin[2] + offset[2],
            ),
        ),
        dtype=np.float32,
    )


def _add_south_face(i: int) -> np.ndarray:
    return np.array(
        (
            3,
            20 + i,
            21 + i,
            22 + i,
            3,
            20 + i,
            23 + i,
            22 + i,
        ),
        dtype=np.uint32,
    )


def _add_north_face(i: int) -> np.ndarray:
    return np.array(
        (
            3,
            0 + i,
            1 + i,
            2 + i,
            3,
            0 + i,
            3 + i,
            2 + i,
        ),
        dtype=np.uint32,
    )


def _add_east_face(i: int) -> np.ndarray:
    return np.array(
        (
            3,
            4 + i,
            5 + i,
            6 + i,
            3,
            4 + i,
            7 + i,
            6 + i,
        ),
        dtype=np.uint32,
    )


def _add_west_face(i: int) -> np.ndarray:
    return np.array(
        (
            3,
            12 + i,
            13 + i,
            14 + i,
            3,
            12 + i,
            15 + i,
            14 + i,
        ),
        dtype=np.uint32,
    )


def _add_face_above(i: int) -> np.ndarray:
    return np.array(
        (
            3,
            8 + i,
            9 + i,
            10 + i,
            3,
            8 + i,
            11 + i,
            10 + i,
        ),
        dtype=np.uint32,
    )


def _add_face_below(i: int) -> np.ndarray:
    return np.array(
        (
            3,
            16 + i,
            17 + i,
            18 + i,
            3,
            16 + i,
            19 + i,
            18 + i,
        ),
        dtype=np.uint32,
    )


def _generate_subchunk_mesh(
    section_origin: tuple[int, int, int],
    palette_data: dict[str, str],
    block_data: list[int],
    single_block_type: bool,
) -> pv.PolyData | None:
    fc_array = []
    pt_array = []
    if single_block_type:
        # Checks if block is one that is not ignored
        if _get_block(palette_data, block_data, 0, 0, 0):
            fc_idx = len(pt_array)
            return pv.PolyData(
                _generate_point_data(16, section_origin),
                faces=np.hstack(
                    (
                        _add_north_face(fc_idx),
                        _add_east_face(fc_idx),
                        _add_face_above(fc_idx),
                        _add_west_face(fc_idx),
                        _add_face_below(fc_idx),
                        _add_south_face(fc_idx),
                    ),
                    dtype=np.uint32,
                ),
            )
    else:
        for y, z, x in iter_product(range(0, 16), range(0, 16), range(0, 16)):
            # Checks if block is one that is not ignored
            if _get_block(palette_data, block_data, x, y, z):
                adjacent_face_data = (
                    _get_block(palette_data, block_data, x, y, z - 1),
                    _get_block(palette_data, block_data, x + 1, y, z),
                    _get_block(palette_data, block_data, x, y + 1, z),
                    _get_block(palette_data, block_data, x - 1, y, z),
                    _get_block(palette_data, block_data, x, y - 1, z),
                    _get_block(palette_data, block_data, x, y, z + 1),
                )
                if not all(adjacent_face_data):
                    fc_index = len(pt_array)
                    pt_array.extend(_generate_point_data(1, section_origin, (x, y, z)))
                    if not adjacent_face_data[0]:
                        fc_array.append(_add_north_face(fc_index))
                    if not adjacent_face_data[1]:
                        fc_array.append(_add_east_face(fc_index))
                    if not adjacent_face_data[2]:
                        fc_array.append(_add_face_above(fc_index))
                    if not adjacent_face_data[3]:
                        fc_array.append(_add_west_face(fc_index))
                    if not adjacent_face_data[4]:
                        fc_array.append(_add_face_below(fc_index))
                    if not adjacent_face_data[5]:
                        fc_array.append(_add_south_face(fc_index))
        return pv.PolyData(pt_array, faces=np.hstack(fc_array, dtype=np.uint32))
    return None
