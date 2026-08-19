__all__ = ["Mesh"]

import pyvista as pv
import numpy as np
from itertools import product as iter_product

import program_references
from coordinate import Coordinate


class Mesh:
    """
    Mesh object containing the mesh data for the chunk.
    """

    def __init__(self, data: pv.PolyData | pv.DataSet | None):
        self.mesh_data = data
        self.is_empty = False if data else True

    @staticmethod
    def generate_subchunk_mesh(subchunk_origin: Coordinate, data, /) -> "Mesh":
        fc_array = []
        pt_array = []

        # Multi block type subchunk
        if "data" in data:
            for y, z, x in iter_product(range(0, 16), range(0, 16), range(0, 16)):
                # Checks if block is one that is not ignored
                if Mesh._block_adjacent(
                    data["palette"], data["data"], Coordinate(x, y, z)
                ):
                    adjacent_face_data = (
                        Mesh._block_adjacent(
                            data["palette"], data["data"], Coordinate(x, y, z - 1)
                        ),
                        Mesh._block_adjacent(
                            data["palette"], data["data"], Coordinate(x + 1, y, z)
                        ),
                        Mesh._block_adjacent(
                            data["palette"], data["data"], Coordinate(x, y + 1, z)
                        ),
                        Mesh._block_adjacent(
                            data["palette"], data["data"], Coordinate(x - 1, y, z)
                        ),
                        Mesh._block_adjacent(
                            data["palette"], data["data"], Coordinate(x, y - 1, z)
                        ),
                        Mesh._block_adjacent(
                            data["palette"], data["data"], Coordinate(x, y, z + 1)
                        ),
                    )
                    if not all(adjacent_face_data):
                        fc_index = len(pt_array)
                        pt_array.extend(
                            Mesh._generate_point_data(
                                subchunk_origin, offset=Coordinate(x, y, z)
                            )
                        )
                        if not adjacent_face_data[0]:
                            fc_array.append(Mesh._add_north_face(fc_index))
                        if not adjacent_face_data[1]:
                            fc_array.append(Mesh._add_east_face(fc_index))
                        if not adjacent_face_data[2]:
                            fc_array.append(Mesh._add_face_above(fc_index))
                        if not adjacent_face_data[3]:
                            fc_array.append(Mesh._add_west_face(fc_index))
                        if not adjacent_face_data[4]:
                            fc_array.append(Mesh._add_face_below(fc_index))
                        if not adjacent_face_data[5]:
                            fc_array.append(Mesh._add_south_face(fc_index))
            if fc_array and pt_array:
                return Mesh(
                    pv.PolyData(pt_array, faces=np.hstack(fc_array, dtype=np.uint32))
                )
        else:
            # Checks if block is one that is not ignored
            if Mesh._block_adjacent(data["palette"], [0], Coordinate(0, 0, 0)):
                fc_idx = len(pt_array)
                return Mesh(
                    pv.PolyData(
                        Mesh._generate_point_data(subchunk_origin, 16),
                        faces=np.hstack(
                            (
                                Mesh._add_north_face(fc_idx),
                                Mesh._add_east_face(fc_idx),
                                Mesh._add_face_above(fc_idx),
                                Mesh._add_west_face(fc_idx),
                                Mesh._add_face_below(fc_idx),
                                Mesh._add_south_face(fc_idx),
                            ),
                            dtype=np.uint32,
                        ),
                    )
                )

        return Mesh(None)

    @staticmethod
    def _block_adjacent(
        palette_data: list[dict[str, str]],
        block_data: np.ndarray,
        coordinate: Coordinate,
    ) -> bool:
        if (
            15 >= coordinate.z >= 0
            and 15 >= coordinate.y >= 0
            and 15 >= coordinate.x >= 0
            and palette_data[block_data[coordinate.get_block_index()]]["Name"]
            not in program_references.blocks_ignore
        ):
            return True
        return False

    @staticmethod
    def _generate_point_data(
        origin: Coordinate,
        mult_factor: int = 1,
        offset: Coordinate = Coordinate(0, 0, 0),
    ) -> np.ndarray:
        return np.array(
            (
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    0.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    0.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    1.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
                (
                    1.0 * mult_factor + origin.x + offset.x,
                    0.0 * mult_factor + origin.y + offset.y,
                    1.0 * mult_factor + origin.z + offset.z,
                ),
            ),
            dtype=np.float32,
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
