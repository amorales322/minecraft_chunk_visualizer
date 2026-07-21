from zlib import decompress as zlib_decompress
from gzip import decompress as gzip_decompress
from lz4.frame import decompress as lz4_decompress
from json import dump
from time import strftime, localtime
from os.path import abspath, commonpath
import pyvista as pv
import numpy as np

import Plotter
from NBTParser import generate_NBT_Data

__all = [
    "Chunk",
    "InvalidCompressionTypeError",
    "FileCorruptionError",
]


class InvalidCompressionTypeError(Exception):
    def __init__(
        self, compression_type: int, message: str = "Unknown compression type"
    ):
        print(f"{message}: {compression_type}")


class FileCorruptionError(Exception):
    def __init__(self, message: str):
        print(f"File corrupted: {message}")


class Chunk:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_region_data(region_data, xpos: int, zpos: int):
        if isinstance(region_data.data, dict):
            return region_data.data[f"Chunk [{xpos % 32}, {zpos % 32}]"]
        else:
            header_offset = 4 * ((xpos % 32) + ((zpos % 32) * 32))
            sector_offset = (
                int.from_bytes(
                    region_data.data[header_offset : header_offset + 3], "big"
                )
                * 4096
            )
            sector_count = int.from_bytes(
                region_data.data[header_offset + 3 : header_offset + 4], "big"
            )
            byte_size = int.from_bytes(
                region_data.data[sector_offset : sector_offset + 4], "big", signed=True
            )
            compression_type = int.from_bytes(
                region_data.data[sector_offset + 4 : sector_offset + 5],
                "big",
            )

            if sector_count == 0:
                return Chunk({})
            match compression_type:
                case 1:
                    # Not tested, may not work
                    return Chunk(
                        generate_NBT_Data(
                            gzip_decompress(
                                region_data.data[
                                    sector_offset + 5 : sector_offset + 5 + byte_size
                                ]
                            )
                        )
                    )
                case 2:
                    return Chunk(
                        generate_NBT_Data(
                            zlib_decompress(
                                region_data.data[
                                    sector_offset + 5 : sector_offset + 5 + byte_size
                                ]
                            ),
                            xpos,
                            zpos,
                        )
                    )
                case 3:
                    # Not tested, may not work
                    return Chunk(
                        generate_NBT_Data(
                            region_data.data[
                                sector_offset + 5 : sector_offset + 5 + byte_size
                            ]
                        )
                    )
                case 4:
                    # Not tested, may not work
                    return Chunk(
                        generate_NBT_Data(
                            lz4_decompress(
                                region_data.data[
                                    sector_offset + 5 : sector_offset + 5 + byte_size
                                ],
                                return_bytearray=True,
                            )
                        )
                    )
                case 127:
                    raise InvalidCompressionTypeError(
                        compression_type,
                        message="Decompressing custom compression algorithms are not supported.",
                    )
                case _:
                    if compression_type >= 128:
                        raise Exception(
                            f"Compression type is {compression_type}. This is for .mcc files which are not supported."
                        )
                    else:
                        raise FileCorruptionError(
                            f"Compression type is not a valid value (Value: {compression_type}). File may be corrupted."
                        )

    def generate_mesh(self) -> list[pv.PolyData]:
        meshes = []
        if self.data:
            for section in self.data["sections"].values():
                if section["Y"] >= -4:
                    palette = section["block_states"]["palette"]
                    if len(palette) != 1:
                        block_buffer = []
                        bit_width = max(len(f"{len(palette) - 1:b}"), 4)
                        for value in section["block_states"]["data"].values():
                            bin_val = np.binary_repr(value, width=64)
                            for v in range(64, 64 % bit_width, -bit_width):
                                block_buffer.append(
                                    palette[str(int(bin_val[v - bit_width : v], 2))][
                                        "Name"
                                    ]
                                )
                        mesh = Plotter.generate_subchunk_mesh(
                            section_origin=(
                                16 * self.data["xPos"],
                                16 * section["Y"],
                                16 * self.data["zPos"],
                            ),
                            block_data=block_buffer,
                            single_block_type=False,
                        )
                        if type(mesh) is pv.PolyData:
                            meshes.append(mesh)
                    else:
                        mesh = Plotter.generate_subchunk_mesh(
                            section_origin=(
                                16 * self.data["xPos"],
                                16 * section["Y"],
                                16 * self.data["zPos"],
                            ),
                            block_data=[palette["0"]["Name"]],
                            single_block_type=True,
                        )
                        if type(mesh) is pv.PolyData:
                            meshes.append(mesh)
        return meshes

    def export_as_JSON(self, path: str) -> None:
        root_file_path = "/" + "/".join(abspath(__file__).split("/")[1:-1])
        file_path = path
        try:
            with open(file_path, "w") as file:
                dump(self.data, file)
        except OSError:
            print("Could not open file. Opening default file.")
            file_path = f"Extracted_NBT_Chunk_Data_{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.json"
            with open(file_path, "w") as file:
                dump(self.data, file)
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
