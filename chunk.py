from json import dump
from time import strftime, localtime
from os.path import abspath, commonpath
import pyvista as pv
import numpy as np

from block import Block
from nbt_parser import NBTData
from coordinate import Coordinate
from mesh import Mesh

__all = ["Chunk"]


class Chunk:
    """
    Chunk class containing the chunk NBT data.
    """

    def __init__(self, data: NBTData) -> None:
        if data.nbt_data and data.nbt_data["Status"] == "minecraft:full":
            self.section_block_counts = {}
            self.section_palette_lengths = {i: 0 for i in range(-4, 20)}
            for section in data.nbt_data["sections"]:
                if section["Y"] >= -4:
                    subchunk_y = section["Y"]
                    print(section["block_states"]["palette"])
                    palette_length = len(section["block_states"]["palette"])
                    self.section_palette_lengths[subchunk_y] = palette_length
                    if palette_length > 1:
                        n_elem = len(section["block_states"]["data"])
                        elem_bit_width = max((palette_length - 1).bit_length(), 4)
                        idx = 0
                        entry_buffer = np.zeros(
                            int(
                                (n_elem * 64 - (n_elem * (64 % elem_bit_width)))
                                / elem_bit_width
                            ),
                            dtype=np.uint16,
                        )

                        for elem in section["block_states"]["data"]:
                            bin_val = np.binary_repr(elem, width=64)
                            for v in range(64, 64 % elem_bit_width, -elem_bit_width):
                                entry_buffer[idx] = int(
                                    bin_val[v - elem_bit_width : v], 2
                                )
                                idx += 1
                        print(section)
                        section["block_states"]["data"] = entry_buffer
                        self._set_block_counts(
                            subchunk_y,
                            self._calculate_block_count(entry_buffer, dtype=np.uint16),
                        )
                    else:
                        self._set_block_counts(
                            subchunk_y,
                            np.array([4096], dtype=np.uint16),
                        )

            self.chunk_data = data.nbt_data
            self.chunk_metadata = data.nbt_structure
            self.is_empty = False
        else:
            self.chunk_data = None
            self.chunk_metadata = None
            self.is_empty = True

    def _get_section_index(self, y_coord: int) -> int:
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return ((y_coord + 64) // 16 - 4) + origin_offset + 4

    def _section_idx_to_section(self, section_idx: int) -> int:
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return section_idx - origin_offset - 4

    def _section_to_section_index(self, section: int) -> int:
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return section + origin_offset + 4

    # For multiple blocks
    def _calculate_block_count(self, data: np.ndarray, dtype=np.int64) -> np.ndarray:
        numbers, n_occurrences = np.unique(data, return_counts=True)
        return n_occurrences.astype(dtype)

    # For single block
    def _update_block_count(
        self, section_y: int, data, original_block_idx: int
    ) -> None:
        # Decrement original block
        self.section_block_counts[section_y][original_block_idx] -= 1

        # Remove original block if zero
        if self.section_block_counts[section_y][original_block_idx] == 0:
            data["data"][data["data"] > original_block_idx] -= 1
            data["palette"].pop(original_block_idx)
            self.section_block_counts[section_y] = np.delete(
                self.section_block_counts[section_y], original_block_idx
            )

    def _set_block_counts(self, section_y: int, data: np.ndarray) -> None:
        self.section_block_counts[section_y] = data

    def _get_block_counts(self, section_y: int) -> np.ndarray:
        return self.section_block_counts[section_y]

    def _get_next_palette_index(self, section_y: int) -> int:
        index = self.section_palette_lengths[section_y]
        self.section_palette_lengths[section_y] += 1
        return index

    def _get_section_data(self, section_y: int):
        return self.chunk_data["sections"][self._section_to_section_index(section_y)]

    def _set_section_data(self, section_y: int, data) -> None:
        self.chunk_data["sections"][self._section_to_section_index(section_y)] = data

    def _contains_multiple_blocks(self, section_y: int) -> bool:
        return (
            "data"
            in self.chunk_data["sections"][self._section_to_section_index(section_y)][
                "block_states"
            ]
        )

    def _block_already_present(self, palette_data, block: Block) -> tuple[bool, int]:
        exists, index = (False, 0)
        for i, palette_item in enumerate(palette_data):
            if block.to_NBT_format() == palette_item:
                exists, index = (True, i)
                break
        return exists, index

    def generate_mesh(self) -> Mesh:
        """
        Generates a pyvista PolyData mesh of the chunk.

        :return: `PolyData` Object containing the mesh data for the chunk, `None` if the chunk only contains blocks found in `program_references.blocks_ignore`
        :rtype: PolyData | None
        """

        meshes = []
        if not self.is_empty:
            for section in self.chunk_data["sections"]:
                if section["Y"] >= -4:
                    mesh = Mesh.generate_subchunk_mesh(
                        Coordinate(
                            16 * self.chunk_data["xPos"],
                            16 * section["Y"],
                            16 * self.chunk_data["zPos"],
                        ),
                        section["block_states"],
                    )
                    if not mesh.is_empty:
                        meshes.append(mesh.mesh_data)
            if meshes:
                return Mesh(pv.merge(meshes))
        return Mesh(None)

    def export_as_JSON(self, path: str) -> None:
        """
        Exports the chunk data as JSON.

        :param path: Filepath to export to.
        :type path: str
        :return: `None`
        :rtype: None
        """
        root_file_path = "/" + "/".join(abspath(__file__).split("/")[1:-1])
        file_path = path
        try:
            with open(file_path, "w") as file:
                dump(self.chunk_data, file)
        except OSError:
            print("Could not open file. Opening default file.")
            file_path = f"Extracted_NBT_Chunk_Data_{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.json"
            with open(file_path, "w") as file:
                dump(self.chunk_data, file)
        finally:
            try:
                common_path = commonpath([abspath(__file__), file_path])
                file_path = common_path + file_path.split(common_path)[-1]
            except ValueError:
                file_path = root_file_path + "/" + file_path
            finally:
                print(
                    f'Chunk data has been exported as JSON and saved to "{file_path}".'
                )

    def get_block(self, coordinate: Coordinate) -> Block:
        """
        Get block at specified position relative to the chunk origin.

        :param coordinate: Coordinate object defining the coordinate of the block to get relative to the chunk origin -> ([0, 15], [-64, 320], [0, 15]).
        :type coordinate: Coordinate
        :return: Block at specified position.
        :rtype: Block
        :raise ValueError: Raised if the X, Y, or Z coordinates are outside the valid range.
        """

        if not 0 <= coordinate.x <= 15:
            raise ValueError(
                f"{coordinate.x} is outside the valid range. X coordinate must be between 0 and 15."
            )
        if not -64 <= coordinate.y <= 320:
            raise ValueError(
                f"{coordinate.y} is outside the valid range. Y coordinate must be between -64 and 320."
            )
        if not 0 <= coordinate.z <= 15:
            raise ValueError(
                f"{coordinate.z} is outside the valid range. Z coordinate must be between 0 and 15."
            )

        section = self._get_section_index(coordinate.y)
        block_idx = coordinate.get_block_index()
        block_id = self.chunk_data["sections"][section]["block_states"]["data"][
            block_idx
        ]
        block = self.chunk_data["sections"][section]["block_states"]["palette"][
            block_id
        ]

        return Block(
            block["Name"], (block["Properties"] if "Properties" in block else {})
        )

    def set_block(self, block: Block, coordinate: Coordinate) -> None:
        """
        Sets the block at the specified position relative to the chunk origin.

        :param block: Block object to set the block to.
        :type block: Block
        :param coordinate: Coordinate object defining the coordinate of the block to modify relative to the chunk origin -> ([0, 15], [-64, 320], [0, 15]).
        :type coordinate: Coordinate
        :return: `None`
        :rtype: None
        :raise ValueError: Raised if the X, Y, or Z coordinates are outside the valid range.
        """
        if not 0 <= coordinate.x <= 15:
            raise ValueError(
                f"{coordinate.x} is outside the valid range. X coordinate must be between 0 and 15."
            )
        if not -64 <= coordinate.y <= 320:
            raise ValueError(
                f"{coordinate.y} is outside the valid range. Y coordinate must be between -64 and 320."
            )
        if not 0 <= coordinate.z <= 15:
            raise ValueError(
                f"{coordinate.z} is outside the valid range. Z coordinate must be between 0 and 15."
            )

        section = coordinate.get_section()
        block_idx = coordinate.get_block_index()
        section_data = self._get_section_data(section)["block_states"]

        if self._contains_multiple_blocks(section):
            original_block = section_data["data"][block_idx]

            # Check if the block is already present inside the chunk
            block_already_present, palette_index = self._block_already_present(
                section_data["palette"], block
            )

            if block_already_present:
                section_data["data"][block_idx] = palette_index
                self.section_block_counts[section][palette_index] += 1
                self._update_block_count(section, section_data, original_block)
            else:
                new_index = self._get_next_palette_index(section)

                # Append new block
                section_data["data"][block_idx] = new_index
                section_data["palette"].append(block.to_NBT_format())
                self._set_block_counts(
                    section, np.append(self.section_block_counts[section], 1)
                )
                self._update_block_count(section, section_data, original_block)
        else:
            if block.to_NBT_format() != section_data["palette"][0]:
                # If blocks are not equal
                section_data["data"] = np.full(4096, 0)
                section_data["data"][block_idx] = 1
                section_data["palette"].append(block.to_NBT_format())
                self.section_block_counts[section] = np.array(
                    [4095, 1], dtype=np.uint16
                )
        self.chunk_data["sections"][self._section_to_section_index(section)][
            "block_states"
        ] = section_data
