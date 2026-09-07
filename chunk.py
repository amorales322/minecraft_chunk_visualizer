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

    Block data for each subchunk is calculated and stored in a one-dimensional numpy array of length 4096 (16 * 16 * 16)
    for easy editing. Only chunks that have been fully generated (status == 'minecraft:full' or status == 'full') return
    non-empty Chunk instances. This can be checked using the 'is_empty' attribute.
    """

    def __init__(self, data: NBTData | None, /) -> None:
        if data is None:
            self.chunk_data = None
            self.chunk_metadata = None
            self.is_empty = True
        elif type(data) is not NBTData:
            raise ValueError("NBT Data has not been provided.")
        else:
            # Used to calculate and index of block counts for each subchunk.
            # Checking for the substring 'full' makes it version independent, as this has changed throughout game versions.
            if "full" in data.nbt_data["Status"]:
                self.block_count_idx = {
                    i: np.zeros(1, dtype=np.uint16) for i in range(-4, 20)
                }
                self.section_palette_len = {i: 0 for i in range(-4, 20)}

                # Iterate over each section
                for section in data.nbt_data["sections"]:
                    # Anything below y = -4 and above y = 19 are outside the normal world height limits, so these subchunks
                    # will be ignored.
                    if section["Y"] >= -4:
                        palette_length = len(section["block_states"]["palette"])
                        self.section_palette_len[section["Y"]] = palette_length

                        if palette_length > 1:
                            n_elem = len(section["block_states"]["data"])
                            elem_bit_width = max((palette_length - 1).bit_length(), 4)
                            block_buf_size = int(
                                (n_elem * 64 - (n_elem * (64 % elem_bit_width)))
                                / elem_bit_width
                            )

                            # Create an entry buffer to fit all blocks inside the array
                            entry_buffer = np.zeros(block_buf_size, dtype=np.uint16)

                            # Iterate over every number and extract the palette indices
                            block_idx = 0
                            for elem in section["block_states"]["data"]:
                                bin_val = np.binary_repr(elem, width=64)
                                for v in range(
                                    64, 64 % elem_bit_width, -elem_bit_width
                                ):
                                    entry_buffer[block_idx] = int(
                                        bin_val[v - elem_bit_width : v], 2
                                    )
                                    block_idx += 1

                            # Save to subchunk data
                            section["block_states"]["data"] = entry_buffer

                            # Calculate block counts
                            self._set_block_counts(
                                self._calculate_block_counts(entry_buffer),
                                section["Y"],
                            )
                        else:
                            self._set_block_counts(
                                np.array([4096], dtype=np.uint16),
                                section["Y"],
                            )

                self.chunk_data = data.nbt_data
                self.chunk_metadata = data.nbt_structure
                self.is_empty = False
            else:
                self.chunk_data = None
                self.chunk_metadata = None
                self.is_empty = True

    @staticmethod
    def get_section_y(y: int) -> int:
        """
        Calculates y position of the subchunk from the supplied y-coordinate.

        :param y: y-coordinate
        :type y: int
        :return: The subchunk number based on the y-coordinate.
        :rtype: int
        """

        return y // 16

    # Get section index for the respective y-coordinate or section y-coordinate
    def _get_section_idx(
        self, /, *, y: int | None = None, section: int | None = None
    ) -> int:
        if not y and not section:
            raise ValueError("Value not supplied.")
        if y:
            section = self.get_section_y(y)
        origin_offset = -4 - self.chunk_data["sections"][0]["Y"]
        return section + origin_offset + 4

    def _calculate_block_counts(self, data: np.ndarray, /) -> np.ndarray:
        numbers, n_occurrences = np.unique(data, return_counts=True)
        return n_occurrences.astype(dtype=np.uint16)

    # When a single block is updated
    def _update_block_count(
        self, /, old_idx: int, new_idx: int, section_y: int
    ) -> None:
        section_idx = self._get_section_idx(section=section_y)

        # If a new block is being added to the chunk, append count for the new index
        if new_idx > self.block_count_idx[section_y].shape[0] - 1:
            self._set_block_counts(
                np.append(self.block_count_idx[section_y], 1),
                section_y,
            )

        # Increment the count for the new block
        self.block_count_idx[section_y][new_idx] += 1

        # Decrement original block
        self.block_count_idx[section_y][old_idx] -= 1

        # Remove original block if zero
        if self.block_count_idx[section_y][old_idx] == 0:
            self.chunk_data["sections"][section_idx]["block_states"]["data"][
                self.chunk_data["sections"][section_idx]["block_states"]["data"]
                > old_idx
            ] -= 1
            self.chunk_data["sections"][section_idx]["block_states"]["palette"].pop(
                old_idx
            )
            self.block_count_idx[section_y] = np.delete(
                self.block_count_idx[section_y], old_idx
            )

    def _set_block_counts(self, data: np.ndarray, section_y: int, /) -> None:
        self.block_count_idx[section_y] = data

    def _get_block_counts(self, section_y: int, /) -> np.ndarray:
        return self.block_count_idx[section_y]

    def _get_next_palette_idx(self, section_y: int, /) -> int:
        idx = self.section_palette_len[section_y]
        self.section_palette_len[section_y] += 1
        return idx

    def _get_section_data(self, section_y: int, /):
        return self.chunk_data["sections"][self._get_section_idx(section=section_y)]

    def _set_section_data(
        self, data: dict[str, dict[str, str] | np.ndarray], section_y: int, /
    ) -> None:
        self.chunk_data["sections"][self._get_section_idx(section=section_y)] = data

    def _contains_multiple_blocks(self, section_y: int, /) -> bool:
        idx = self._get_section_idx(section=section_y)
        return "data" in self.chunk_data["sections"][idx]["block_states"]

    def _block_already_present(
        self, palette_data: dict[str, dict[str, str] | str], block: Block, /
    ) -> tuple[bool, int]:
        exists, idx = (False, 0)
        for i, palette_item in enumerate(palette_data):
            if block.to_NBT_format() == palette_item:
                exists, idx = (True, i)
                break
        return exists, idx

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

    def export_as_JSON(self, path: str, /) -> None:
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

    def get_block(self, coordinate: Coordinate, /) -> Block:
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

        section = self._get_section_idx(y=coordinate.y)
        block_block_idx = coordinate.get_block_index()
        block_id = self.chunk_data["sections"][section]["block_states"]["data"][
            block_block_idx
        ]
        block = self.chunk_data["sections"][section]["block_states"]["palette"][
            block_id
        ]

        return Block(
            block["Name"], (block["Properties"] if "Properties" in block else {})
        )

    def set_block(self, block: Block, coordinate: Coordinate, /) -> None:
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

        section = self.get_section_y(y=coordinate.y)
        block_idx = coordinate.get_block_index()
        section_data = self._get_section_data(section)["block_states"]

        if self._contains_multiple_blocks(section):
            original_block_idx = section_data["data"][block_idx]

            # Check if the block is already present inside the chunk
            block_exists, palette_idx = self._block_already_present(
                section_data["palette"], block
            )

            if block_exists:
                section_data["data"][block_idx] = palette_idx
                self._update_block_count(original_block_idx, palette_idx, section)
            else:
                new_idx = self._get_next_palette_idx(section)

                # Append new block
                section_data["data"][block_idx] = new_idx
                section_data["palette"].append(block.to_NBT_format())
                self._update_block_count(original_block_idx, new_idx, section)
        else:
            if block.to_NBT_format() != section_data["palette"][0]:
                # If blocks are not equal
                section_data["data"] = np.full(4096, 0)
                section_data["data"][block_idx] = 1
                section_data["palette"].append(block.to_NBT_format())
                self.block_count_idx[section] = np.array([4095, 1], dtype=np.uint16)
        self.chunk_data["sections"][self._get_section_idx(section=section)][
            "block_states"
        ] = section_data
