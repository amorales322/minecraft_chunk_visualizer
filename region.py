from zlib import decompress as zlib_decompress
from gzip import decompress as gzip_decompress
from lz4.frame import decompress as lz4_decompress
from itertools import product as iter_product
from datetime import datetime, timezone
from math import floor, ceil

import numpy as np

from chunk import Chunk
from nbt_parser import (
    NBTData,
    BYTE_UNSIGNED_DTYPE,
    SHORT_UNSIGNED_DTYPE,
    INT_UNSIGNED_DTYPE,
)

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
    """

    def __init__(
        self,
        data: bytes,
    ):
        self.header_offset = np.zeros(1024, dtype=SHORT_UNSIGNED_DTYPE)
        self.data_byte_offset = np.zeros(1024, dtype=INT_UNSIGNED_DTYPE)
        self.data_sector_size = np.zeros(1024, dtype=BYTE_UNSIGNED_DTYPE)
        self.timestamps = np.zeros(1024, dtype=INT_UNSIGNED_DTYPE)
        for x, z in iter_product(range(0, 32), range(0, 32)):
            header_offset = 4 * (x + (z * 32))
            byte_offset = (
                int.from_bytes(data[header_offset : header_offset + 3], "big") * 4096
            )
            sector_count = int.from_bytes(
                data[header_offset + 3 : header_offset + 4], "big"
            )
            timestamp = int.from_bytes(
                data[header_offset + 4096 : header_offset + 4100], "big"
            )
            self.header_offset[(x * 32) + z] = header_offset
            self.data_byte_offset[(x * 32) + z] = byte_offset
            self.data_sector_size[(x * 32) + z] = sector_count
            self.timestamps[(x * 32) + z] = timestamp


class ChunkMetadata:
    """
    Class containing the chunk metadata.
    """

    def __init__(self):
        self.byte_size = np.zeros(1024, dtype=INT_UNSIGNED_DTYPE)
        self.compression_type = np.zeros(1024, dtype=BYTE_UNSIGNED_DTYPE)


class Region:
    """
    Region class containing the region file data.
    """

    def __init__(self, data: bytes):
        self.header_metadata = HeaderMetadata(data[0:8192])
        self.chunk_metadata = ChunkMetadata()
        self.data = []

        for x, z in iter_product(range(0, 32), range(0, 32)):
            sector_offset = self.header_metadata.data_byte_offset[x * 32 + z]
            byte_size = int.from_bytes(
                data[sector_offset : sector_offset + 4], "big", signed=True
            )
            compression_type = int.from_bytes(
                data[sector_offset + 4 : sector_offset + 5],
                "big",
            )

            if self.header_metadata.data_sector_size[x * 32 + z] == 0:
                self.data.append(None)
            else:
                self.chunk_metadata.byte_size[x * 32 + z] = byte_size
                self.chunk_metadata.compression_type[x * 32 + z] = compression_type
                self.data.append(
                    data[sector_offset + 5 : sector_offset + 5 + byte_size]
                )

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

    def write_chunk(self, data: Chunk, x: int, z: int, /) -> None:
        serialized_data = NBTData.serialize(data)
        bytesize = len(serialized_data)
        sector_count = ceil(bytesize / 4096)
        init_sector_size = np.astype(
            self.header_metadata.data_sector_size[x * 32 + z], np.uint64
        )
        init_sector_offset = self.header_metadata.data_byte_offset[x * 32 + z]

        self.header_metadata.timestamps[(x * 32) + z] = floor(
            datetime.now(timezone.utc).timestamp()
        )
        self.chunk_metadata.byte_size[x * 32 + z] = bytesize
        self.header_metadata.data_sector_size[x * 32 + z] = sector_count

        ## Update sector offsets
        if sector_count > init_sector_size:
            diff = sector_count - init_sector_size
            self.header_metadata.data_byte_offset[
                self.header_metadata.data_byte_offset > init_sector_offset
            ] += diff * 4096
        elif sector_count < init_sector_size:
            diff = init_sector_size - sector_count
            self.header_metadata.data_byte_offset[
                self.header_metadata.data_byte_offset > init_sector_offset
            ] -= diff * 4096

        serialized_data += b"\0" * (4096 - (len(serialized_data) % 4096))
        self.data[x * 32 + z] = serialized_data

    def get_chunk(self, x: int, z: int, /) -> Chunk | None:
        """
        Returns a Chunk object containing the modifiable chunk data.

        :param x: X-coordinate of chunk relative to region origin -> [0, 31].
        :type x: int
        :param z: Z-coordinate of chunk relative to region origin -> [0, 31].
        :type z: int
        :return: `Chunk` object containing the modifiable chunk data, or `None` if the chunk has not been generated.
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
        idx = x * 32 + z
        compression_type = self.chunk_metadata.compression_type[idx]

        if not self.data[idx]:
            return None
        match compression_type:
            case 1:
                decompressed_data = gzip_decompress(self.data[idx])
            case 2:
                decompressed_data = zlib_decompress(self.data[idx])
            case 3:
                decompressed_data = self.data[idx]
            case 4:
                decompressed_data = lz4_decompress(
                    self.data[idx],
                    return_bytearray=True,
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
