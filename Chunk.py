from json import dump
from time import strftime, localtime
from os.path import abspath, commonpath
from itertools import product as iter_product
import pyvista as pv
import numpy as np

import Plotter
from block import Block
from NBTParser import NBTData


__all = ["Chunk"]


class Chunk:
    """
    Chunk class containing the chunk NBT data.
    """

    def __init__(self, data: NBTData) -> None:
        if data.nbt_data["Status"] == "minecraft:full":
            self.section_block_counts = {}
            self.section_palette_lengths = {i: 0 for i in range(-4, 20)}
            for section in data.nbt_data["sections"]:
                if section["Y"] >= -4:
                    palette = section["block_states"]["palette"]
                    palette_length = len(palette)
                    self.section_palette_lengths[section["Y"]] = palette_length
                    if palette_length != 1:
                        n_elem = len(section["block_states"]["data"])

                        bit_width = max(palette_length.bit_length(), 4)
                        buffer = np.zeros(
                            int(
                                (n_elem * 64 - (n_elem * (64 % bit_width))) / bit_width
                            ),
                            dtype=np.uint16,
                        )
                        block_idx = 0
                        for value in section["block_states"]["data"]:
                            bin_val = np.binary_repr(value, width=64)
                            for v in range(64, 64 % bit_width, -bit_width):
                                buffer[block_idx] = int(bin_val[v - bit_width : v], 2)
                                block_idx += 1
                        section["block_states"]["data"] = buffer
                        self.section_block_counts[section["Y"]] = (
                            self._calculate_block_index(buffer, dtype=np.uint16)
                        )
                    else:
                        self.section_block_counts[section["Y"]] = np.array(
                            [4096], dtype=np.uint16
                        )

            self.chunk_data = data.nbt_data
            self.chunk_metadata = data.nbt_structure
        else:
            self.chunk_data = None
            self.chunk_metadata = None

    def _get_section_index(self, y_coord: int) -> int:
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return ((y_coord + 64) // 16 - 4) + origin_offset + 4

    @staticmethod
    def _get_section(y_coord: int) -> int:
        return (y_coord + 64) // 16 - 4

    @staticmethod
    def _get_section_relative_y(y: int) -> int:
        return (y + 64) % 16

    def _section_index_to_section(self, section_idx: int) -> int:
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return section_idx - origin_offset - 4

    def _section_to_section_index(self, section: int) -> int:
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return section + origin_offset + 4

    @staticmethod
    def _get_block_index(x: int, y: int, z: int) -> int:
        return ((y + 64) % 16) * 256 + z * 16 + x

    # For multiple blocks
    def _calculate_block_index(self, data: np.ndarray, dtype=np.int64) -> np.ndarray:
        numbers, n_occurrences = np.unique(data, return_counts=True)
        return n_occurrences.astype(dtype)

    # For single block
    def _update_block_index(
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

    def _get_next_palette_index(self, section: int) -> int:
        section = section
        index = self.section_palette_lengths[section]
        self.section_palette_lengths[section] += 1
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
            if block._to_NBT_format() == palette_item:
                exists, index = (True, i)
                break
        return exists, index

    def generate_mesh(self) -> pv.PolyData | None:
        """
        Generates a pyvista PolyData mesh of the chunk.

        :return: `PolyData` Object containing the mesh data for the chunk, `None` if the chunk only contains blocks found in `program_references.blocks_ignore`
        :rtype: PolyData | None
        """
        meshes = []
        if self.chunk_data:
            for section in self.chunk_data["sections"]:
                if section["Y"] >= -4:
                    if len(section["block_states"]["palette"]) != 1:
                        mesh = Plotter._generate_subchunk_mesh(
                            section_origin=(
                                16 * self.chunk_data["xPos"],
                                16 * section["Y"],
                                16 * self.chunk_data["zPos"],
                            ),
                            palette_data=section["block_states"]["palette"],
                            block_data=section["block_states"]["data"],
                            single_block_type=False,
                        )
                        if type(mesh) is pv.PolyData:
                            meshes.append(mesh)
                    else:
                        mesh = Plotter._generate_subchunk_mesh(
                            section_origin=(
                                16 * self.chunk_data["xPos"],
                                16 * section["Y"],
                                16 * self.chunk_data["zPos"],
                            ),
                            palette_data=section["block_states"]["palette"],
                            block_data=[0],
                            single_block_type=True,
                        )
                        if type(mesh) is pv.PolyData:
                            meshes.append(mesh)
        if meshes:
            return pv.merge(meshes)
        return None

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

    def get_block(self, x_rel: int, y_rel: int, z_rel: int) -> Block:
        """
        Get block at specified position relative to the chunk origin.

        :param x_rel: X Coordinate relative to the chunk origin -> [0, 15].
        :type x_rel: int
        :param y_rel: Y Coordinate relative to the chunk origin -> [0, 15].
        :type y_rel: int
        :param z_rel: Z Coordinate relative to the chunk origin -> [-64, 320].
        :type z_rel: int
        :return: Block at specified position.
        :rtype: Block
        :raise ValueError: Raised if the X, Y, or Z coordinates are outside the valid range.
        """

        if not 0 <= x_rel <= 15:
            raise ValueError(
                f"{x_rel} is outside the valid range. X coordinate must be between 0 and 15."
            )
        if not 0 <= z_rel <= 15:
            raise ValueError(
                f"{z_rel} is outside the valid range. Z coordinate must be between 0 and 15."
            )
        if not -64 <= y_rel <= 320:
            raise ValueError(
                f"{y_rel} is outside the valid range. Y coordinate must be between -64 and 320."
            )

        section = self._get_section_index(y_rel)
        block_idx = self._get_block_index(x_rel, y_rel, z_rel)
        block_id = self.chunk_data["sections"][section]["block_states"]["data"][
            block_idx
        ]
        block = self.chunk_data["sections"][section]["block_states"]["palette"][
            block_id
        ]

        return Block(
            block["Name"], (block["Properties"] if "Properties" in block else {})
        )

    def set_block(self, block: Block, x_rel: int, y_rel: int, z_rel: int) -> None:
        """
        Sets the block at the specified position relative to the chunk origin.

        :param block: Block object to set the block to.
        :type block: Block
        :param x_rel: X Coordinate relative to the chunk origin -> [0, 15].
        :type x_rel: int
        :param y_rel: Y Coordinate relative to the chunk origin -> [0, 15].
        :type y_rel: int
        :param z_rel: Z Coordinate relative to the chunk origin -> [-64, 320].
        :type z_rel: int
        :return: `None`
        :rtype: None
        :raise ValueError: Raised if the X, Y, or Z coordinates are outside the valid range.
        """
        if not 0 <= x_rel <= 15:
            raise ValueError(
                f"{x_rel} is outside the valid range. X coordinate must be between 0 and 15."
            )
        if not 0 <= z_rel <= 15:
            raise ValueError(
                f"{z_rel} is outside the valid range. Z coordinate must be between 0 and 15."
            )
        if not -64 <= y_rel <= 320:
            raise ValueError(
                f"{y_rel} is outside the valid range. Y coordinate must be between -64 and 320."
            )
        section = self._get_section(y_rel)
        block_idx = self._get_block_index(x_rel, y_rel, z_rel)
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
                self._update_block_index(section, section_data, original_block)
            else:
                new_index = self._get_next_palette_index(section)

                # Append new block
                section_data["data"][block_idx] = new_index
                section_data["palette"].append(block._to_NBT_format())
                self._set_block_counts(
                    section, np.append(self.section_block_counts[section], 1)
                )
                self._update_block_index(section, section_data, original_block)
        else:
            if block._to_NBT_format() != section_data["palette"][0]:
                # If blocks are not equal
                section_data["data"] = np.full(4096, 0)
                section_data["data"][block_idx] = 1
                section_data["palette"].append(block._to_NBT_format())
                self.section_block_counts[section] = np.array(
                    [4095, 1], dtype=np.uint16
                )
        self.chunk_data["sections"][self._section_to_section_index(section)][
            "block_states"
        ] = section_data
