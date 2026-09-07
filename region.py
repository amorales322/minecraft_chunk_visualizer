from zlib import decompress as zlib_decompress
from gzip import decompress as gzip_decompress
from lz4.frame import decompress as lz4_decompress
from itertools import product as iter_product


import numpy as np

from chunk import Chunk
from nbt_parser import NBTData

__all__ = [
    "Region",
    "InvalidCompressionTypeError",
    "FileCorruptionError",
    "HeaderMetadata",
    "ChunkMetadata",
]


class InvalidCompressionTypeError(Exception):
    def __init__(
        self, compression_type: int, message: str = "Unknown compression type"
    ):
        print(f"{message}: {compression_type}")


class FileCorruptionError(Exception):
    def __init__(self, message: str):
        print(f"File corrupted: {message}")


class HeaderMetadata:
    """
    Class containing the header metadata.

    Using the format to store the chunk data inside the Region class. Header metadata for each chunk is stored sequentially
    in a one-dimensional array. Chunk metadata for a chunk at a specific coordinate can be access by calculating
    the index using the Region.get_chunk_index static method and passing that index to the respective metadata entry.
    """

    def __init__(
        self,
        data: bytes,
    ):
        self.data_sector_offset = np.zeros(1024, dtype=np.uint32)
        self.data_sector_size = np.zeros(1024, dtype=np.uint8)
        self.timestamps = np.zeros(1024, dtype=np.uint32)
        for z, x in iter_product(range(0, 32), range(0, 32)):
            header_offset = Region._calculate_header_file_offset(x, z)
            idx = Region.get_chunk_index(x, z)

            sector_offset = int.from_bytes(
                data[header_offset : header_offset + 3], "big"
            )
            sector_count = int.from_bytes(
                data[header_offset + 3 : header_offset + 4],
                "big",
            )
            timestamp = int.from_bytes(
                data[header_offset + 4096 : header_offset + 4100], "big"
            )

            if sector_count > 0:
                self.data_sector_offset[idx] = sector_offset
                self.data_sector_size[idx] = sector_count
                self.timestamps[idx] = timestamp


class ChunkMetadata:
    """
    Class containing the chunk metadata.
    """

    def __init__(self):
        self.byte_size = np.zeros(1024, dtype=np.uint32)
        self.compression_type = np.zeros(1024, dtype=np.uint8)


class Region:
    """
    Region class containing the region file data.
    """

    def __init__(self, raw_file_data: bytes):
        self.header_metadata = HeaderMetadata(raw_file_data[0:8192])
        self.chunk_metadata = ChunkMetadata()
        self.data: None | bytes = [None] * 1024

        for z, x in iter_product(range(0, 32), range(0, 32)):
            idx = self.get_chunk_index(x, z)
            byte_offset = self.header_metadata.data_sector_offset[idx] * 4096

            # Get chunk properties
            byte_size = int.from_bytes(
                raw_file_data[byte_offset : byte_offset + 4], "big"
            )
            compression_type = int.from_bytes(
                raw_file_data[byte_offset + 4 : byte_offset + 5],
                "big",
            )

            # If chunk exists
            if self.header_metadata.data_sector_size[idx] > 0:
                self.chunk_metadata.byte_size[idx] = byte_size
                self.chunk_metadata.compression_type[idx] = compression_type
                self.data[idx] = raw_file_data[
                    byte_offset + 5 : byte_offset + 5 + byte_size
                ]

    @staticmethod
    def get_chunk_index(x: int, z: int) -> int:
        return (x * 32) + z

    @staticmethod
    def _calculate_header_file_offset(x: int, z: int) -> int:
        return 4 * (x + z * 32)

    @staticmethod
    def from_region_file(filepath: str, /) -> "Region":
        """
        Returns a Region object containing the region file data.

        :param filepath: Region file path
        :type filepath: str
        :return: 'Region' object containing the region file data.
        :rtype: Region
        :raise ValueError: If the file type is unknown/invalid.
        :raise NotImplementedError: If an .mcr or .mcc file is passed.
        """

        match filepath.split(".")[-1]:
            case "mca":
                with open(filepath, "rb") as file:
                    data = Region(file.read())
                return data
            case "mcr":
                raise NotImplementedError(
                    ".mcr region files (pre-12w07a) are not currently supported."
                )
            case "mcc":
                raise NotImplementedError(".mcc files are not currently supported.")
            case _:
                raise ValueError("Unknown/invalid File Type")

    def get_chunk(self, x: int, z: int, /) -> Chunk:
        """
        Returns a Chunk object containing the modifiable chunk data.

        :param x: X-coordinate of chunk relative to region origin -> [0, 31].
        :type x: int
        :param z: Z-coordinate of chunk relative to region origin -> [0, 31].
        :type z: int
        :return: `Chunk` object containing the modifiable chunk data, or an empty Chunk object if the chunk does not exist.
        :rtype: Chunk | None
        :raise ValueError: If the x or z coordinates are outside the valid range.
        :raise InvalidCompressionTypeError: If the file compression is not a valid compression type or if the file is compressed using a custom compression algorithm (since 24w05a).
        :raise NotImplementedError: If the compression type value is > 128 (.mcc file)
        """
        if not 0 <= x <= 31:
            raise ValueError(
                f"{x} is outside the valid range. Chunk x-coordinate must be between 0 and 31."
            )
        if not 0 <= z <= 31:
            raise ValueError(
                f"{z} is outside the valid range. Chunk z-coordinate must be between 0 and 31."
            )
        idx = self.get_chunk_index(x, z)
        compression_type = self.chunk_metadata.compression_type[idx]

        if not self.data[idx]:
            return Chunk(None)
        match compression_type:
            case 1:
                decompressed_data: bytes = gzip_decompress(self.data[idx])
            case 2:
                decompressed_data: bytes = zlib_decompress(self.data[idx])
            case 3:
                decompressed_data: bytes = self.data[idx]
            case 4:
                decompressed_data: bytes = lz4_decompress(
                    self.data[idx], return_bytearray=True
                )
            case 127:
                raise InvalidCompressionTypeError(
                    compression_type,
                    message="Custom compression algorithms (since 24w05a) are not supported.",
                )
            case _:
                if compression_type >= 128:
                    raise NotImplementedError(".mcc files are not currently supported.")
                else:
                    raise InvalidCompressionTypeError(
                        compression_type,
                        "Compression type is not a valid value. File may be corrupted.",
                    )
        return Chunk(NBTData.read(decompressed_data))
