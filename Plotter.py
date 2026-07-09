import pyvista as pv
import numpy as np
from typing import Sequence

import _GlobalReferences
import _TriangleCulling


__all__ = ["plot", "gen_section_mesh"]


def plot(pt: Sequence[float], fc: Sequence[int]) -> None:
    mesh = pv.PolyData(np.array(pt), faces=np.array(fc))
    mesh.texture_map_to_plane(inplace=True)
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True)
    pl.camera_position = "zx"
    pl.window_size = [1920, 1080]
    pl.add_axes(
        interactive=True,
        line_width=5,
    )
    pl.show()


def _add_south_face(face_index: int, faces: Sequence[int]) -> None:
    faces.extend(
        [
            3,
            20 + face_index,
            21 + face_index,
            22 + face_index,
            3,
            20 + face_index,
            23 + face_index,
            22 + face_index,
        ]
    )


def _add_north_face(face_index: int, faces: Sequence[int]) -> None:
    faces.extend(
        [
            3,
            0 + face_index,
            1 + face_index,
            2 + face_index,
            3,
            0 + face_index,
            3 + face_index,
            2 + face_index,
        ]
    )


def _add_east_face(face_index: int, faces: Sequence[int]) -> None:
    faces.extend(
        [
            3,
            4 + face_index,
            5 + face_index,
            6 + face_index,
            3,
            4 + face_index,
            7 + face_index,
            6 + face_index,
        ]
    )


def _add_west_face(face_index: int, faces: Sequence[int]) -> None:
    faces.extend(
        [
            3,
            12 + face_index,
            13 + face_index,
            14 + face_index,
            3,
            12 + face_index,
            15 + face_index,
            14 + face_index,
        ]
    )


def _add_face_above(face_index: int, faces: Sequence[int]) -> None:
    faces.extend(
        [
            3,
            8 + face_index,
            9 + face_index,
            10 + face_index,
            3,
            8 + face_index,
            11 + face_index,
            10 + face_index,
        ]
    )


def _add_face_below(face_index: int, faces: Sequence[int]) -> None:
    faces.extend(
        [
            3,
            16 + face_index,
            17 + face_index,
            18 + face_index,
            3,
            16 + face_index,
            19 + face_index,
            18 + face_index,
        ]
    )


def gen_section_mesh(
    section_origin: tuple[int, int, int],
    block_data: list[str],
    points: Sequence[float],
    faces: Sequence[int],
    diag_data: list[list[int]],
) -> None:
    # Subchunk with one block type
    if len(set(block_data)) == 1:
        face_index = 0 if not points else len(points)
        if block_data[0] not in _GlobalReferences.blocks_ignore:
            points.extend(
                [
                    [
                        0.0 + section_origin[0],
                        0.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        16.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        16.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        0.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        0.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        0.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        16.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        16.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        16.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        16.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        16.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        16.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        16.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        16.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        0.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        0.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        0.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        0.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        0.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        0.0 + section_origin[1],
                        0.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        0.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        0.0 + section_origin[0],
                        16.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        16.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                    [
                        16.0 + section_origin[0],
                        0.0 + section_origin[1],
                        16.0 + section_origin[2],
                    ],
                ]
            )
            for i in range(0, 21, 4):
                faces.extend(
                    [3, i + face_index, i + 1 + face_index, i + 2 + face_index]
                )
                faces.extend(
                    [3, i + face_index, i + 3 + face_index, i + 2 + face_index]
                )
    else:
        for y in range(0, 16):
            for z in range(0, 16):
                for x in range(0, 16):
                    if (
                        block_data[y * 256 + z * 16 + x]
                        not in _GlobalReferences.blocks_ignore
                    ):
                        adjacent_face_data = [
                            _TriangleCulling.is_block_north(block_data, x, y, z),
                            _TriangleCulling.is_block_east(block_data, x, y, z),
                            _TriangleCulling.is_block_above(block_data, x, y, z),
                            _TriangleCulling.is_block_west(block_data, x, y, z),
                            _TriangleCulling.is_block_below(block_data, x, y, z),
                            _TriangleCulling.is_block_south(block_data, x, y, z),
                        ]
                        if not all(adjacent_face_data):
                            face_index = 0 if not points else len(points)
                            points.extend(
                                [
                                    [
                                        0.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        0.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        0.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        1.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                    [
                                        1.0 + section_origin[0] + x,
                                        0.0 + section_origin[1] + y,
                                        1.0 + section_origin[2] + z,
                                    ],
                                ]
                            )
                            if not adjacent_face_data[0]:
                                _add_north_face(face_index, faces)
                            if not adjacent_face_data[1]:
                                _add_east_face(face_index, faces)
                            if not adjacent_face_data[2]:
                                _add_face_above(face_index, faces)
                            if not adjacent_face_data[3]:
                                _add_west_face(face_index, faces)
                            if not adjacent_face_data[4]:
                                _add_face_below(face_index, faces)
                            if not adjacent_face_data[5]:
                                _add_south_face(face_index, faces)
        diag_data[0][len(diag_data[0]) - 1] = len(points)
        diag_data[1][len(diag_data[1]) - 1] = len(faces)
