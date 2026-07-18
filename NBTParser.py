__all__ = (
    [
        "Chunk",
        "Region",
        "InvalidCompressionTypeError",
        "FileCorruptionError",
        "generate_NBT_Data",
    ],
)

from struct import unpack
from typing import Any
from itertools import product as iter_product
from zlib import decompress as zlib_decompress
from gzip import decompress as gzip_decompress
from lz4.frame import decompress as lz4_decompress
from json import dump
from time import strftime, localtime
from os.path import abspath, commonpath


class InvalidCompressionTypeError(Exception):
    def __init__(
        self, compression_type: int, message: str = "Unknown compression type"
    ):
        print(f"{message}: {compression_type}")


class FileCorruptionError(Exception):
    def __init__(self, message: str):
        print(f"File corrupted: {message}")


# To-Do: Add NBT Parsing for region files
class Region:
    def __init__(self, data: bytes):
        self.data = data

    @staticmethod
    def from_region_file(filepath: str, parse_as_json: bool = False):
        if ".mca" in filepath:
            with open(filepath, "rb") as file:
                data = Region(file.read())
            if parse_as_json:
                data_buffer = Region({})
                for x, z in iter_product(range(0, 32), range(0, 32)):
                    data_buffer.data[f"Chunk [{x}, {z}]"] = Chunk.from_region_data(
                        data, x, z
                    )
                data = data_buffer
            return data
        elif ".mcr" in filepath:
            NotImplementedError(
                ".mcr region files ( < 12w07a (Java Version 1.9 snapshot) are not currently supported."
            )
        elif ".dat" in filepath:
            NotImplementedError(".dat files are not currently supported.")
        elif ".nbt" in filepath:
            NotImplementedError(".nbt files are not currently supported.")

        raise Exception("Unknown File Type")


class Chunk:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_region_data(region_data: Region, xpos: int, zpos: int):
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


def generate_NBT_Data(data: bytes, xpos: int = 0, zpos: int = 0) -> dict[str, Any]:
    NBT_data = {}
    tag_name, tag_name_length, i = _extract_tag_name(data, 0)

    try:
        while i < len(data):
            tag = data[i : i + 1]
            match tag:
                case b"\x00":
                    return NBT_data
                case b"\x01":
                    i = _tag_Byte(data, i, NBT_data, False)
                case b"\x02":
                    i = _tag_Short(data, i, NBT_data, False)
                case b"\x03":
                    i = _tag_Int(data, i, NBT_data, False)
                case b"\x04":
                    i = _tag_Long(data, i, NBT_data, False)
                case b"\x05":
                    i = _tag_Float(data, i, NBT_data, False)
                case b"\x06":
                    i = _tag_Double(data, i, NBT_data, False)
                case b"\x07":
                    i = _tag_Byte_Array(data, i, NBT_data, False)
                case b"\x08":
                    i = _tag_String(data, i, NBT_data, False)
                case b"\x09":
                    i = _tag_List(data, i, NBT_data, False)
                case b"\x0a":
                    i = _tag_Compound(data, i, NBT_data, False)
                case b"\x0b":
                    i = _tag_Int_Array(data, i, NBT_data, False)
                case b"\x0c":
                    i = _tag_Long_Array(data, i, NBT_data, False)
                case _:
                    raise Exception(f"Unknown Tag: {tag}")
    except UnicodeDecodeError as e:
        print(f"Corrupted Chunk Data: Skipping Chunk [{xpos}, {zpos}]")
    return NBT_data


def _extract_tag_name(data: bytes, i: int) -> tuple[str, int, int]:
    tag_name_length = int.from_bytes(data[i + 1 : i + 3], "big")
    tag_name = (
        data[i + 3 : i + 3 + tag_name_length].decode()
        if tag_name_length != 0
        else str(i)
    )

    return tag_name, tag_name_length, i + 3 + tag_name_length


def _tag_Byte(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name
    tag_data = int.from_bytes(data[i : i + 1], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 1


def _tag_Short(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name
    tag_data = int.from_bytes(data[i : i + 2], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 2


def _tag_Int(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name
    tag_data = int.from_bytes(data[i : i + 4], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 4


def _tag_Long(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name
    tag_data = int.from_bytes(data[i : i + 8], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 8


def _tag_Float(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name
    tag_data = unpack(">f", data[i : i + 4])
    NBT_data[tag_name] = tag_data
    return i + 4


def _tag_Double(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name
    tag_data = unpack(">d", data[i : i + 8])
    NBT_data[tag_name] = tag_data
    return i + 8


def _tag_String(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name

    tag_data_length = int.from_bytes(data[i : i + 2], "big")
    tag_data = data[i + 2 : i + 2 + tag_data_length].decode()
    NBT_data[tag_name] = tag_data

    return i + 2 + tag_data_length


def _tag_Byte_Array(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    byte_array = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name

    array_size = int.from_bytes(data[i : i + 4], "big", signed=True)
    i = i + 4

    for x in range(0, array_size, 1):
        tag_data = int.from_bytes(data[i : i + 1], "big", signed=True)
        byte_array[str(x)] = tag_data
        i = i + 1

    NBT_data[tag_name] = byte_array
    return i


def _tag_Int_Array(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    int_array = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name

    array_size = int.from_bytes(data[i : i + 4], "big", signed=True)
    i = i + 4

    for x in range(0, array_size, 1):
        tag_data = int.from_bytes(data[i : i + 4], "big", signed=True)
        int_array[str(x)] = tag_data
        i = i + 4

    NBT_data[tag_name] = int_array
    return i


def _tag_Long_Array(
    data: bytes,
    i: int,
    NBT_data: dict[str, Any],
    root_is_list,
    name: str | None = None,
) -> int:
    long_array = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name

    array_size = int.from_bytes(data[i : i + 4], "big", signed=True)
    i = i + 4

    for x in range(0, array_size, 1):
        tag_data = int.from_bytes(data[i : i + 8], "big", signed=True)
        long_array[str(x)] = tag_data
        i = i + 8

    NBT_data[tag_name] = long_array
    return i


def _tag_List(
    data: bytes,
    i: int,
    root_NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    NBT_data = {}
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name

    tag_ID = data[i : i + 1]
    array_size = int.from_bytes(data[i + 1 : i + 5], "big", signed=True)
    i = i + 1 + 4

    if array_size == 0:
        root_NBT_data[tag_name] = {}
        return i

    for x in range(0, array_size, 1):
        match tag_ID:
            case b"\x00":
                root_NBT_data[tag_name] = NBT_data
                return i + 1
            case b"\x01":
                i = _tag_Byte(data, i, NBT_data, True, str(x))
            case b"\x02":
                i = _tag_Short(data, i, NBT_data, True, str(x))
            case b"\x03":
                i = _tag_Int(data, i, NBT_data, True, str(x))
            case b"\x04":
                i = _tag_Long(data, i, NBT_data, True, str(x))
            case b"\x05":
                i = _tag_Float(data, i, NBT_data, True, str(x))
            case b"\x06":
                i = _tag_Double(data, i, NBT_data, True, str(x))
            case b"\x07":
                i = _tag_Byte_Array(data, i, NBT_data, True, str(x))
            case b"\x08":
                i = _tag_String(data, i, NBT_data, True, str(x))
            case b"\x09":
                i = _tag_List(data, i, NBT_data, True, str(x))
            case b"\x0a":
                i = _tag_Compound(data, i, NBT_data, True, str(x))
            case b"\x0b":
                i = _tag_Int_Array(data, i, NBT_data, True, str(x))
            case b"\x0c":
                i = _tag_Long_Array(data, i, NBT_data, True, str(x))
            case _:
                raise Exception("Unknown Tag")

    root_NBT_data[tag_name] = NBT_data
    return i


def _tag_Compound(
    data: bytes,
    i: int,
    root_NBT_data: dict[str, Any],
    root_is_list: bool,
    name: str | None = None,
) -> int:
    NBT_data = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i) if name is None else name

    x = 0
    while True:
        tag = data[i : i + 1]
        match tag:
            case b"\x00":
                root_NBT_data[tag_name] = NBT_data
                return i + 1
            case b"\x01":
                i = _tag_Byte(data, i, NBT_data, False)
            case b"\x02":
                i = _tag_Short(data, i, NBT_data, False)
            case b"\x03":
                i = _tag_Int(data, i, NBT_data, False)
            case b"\x04":
                i = _tag_Long(data, i, NBT_data, False)
            case b"\x05":
                i = _tag_Float(data, i, NBT_data, False)
            case b"\x06":
                i = _tag_Double(data, i, NBT_data, False)
            case b"\x07":
                i = _tag_Byte_Array(data, i, NBT_data, False)
            case b"\x08":
                i = _tag_String(data, i, NBT_data, False)
            case b"\x09":
                i = _tag_List(data, i, NBT_data, False)
            case b"\x0a":
                i = _tag_Compound(data, i, NBT_data, False)
            case b"\x0b":
                i = _tag_Int_Array(data, i, NBT_data, False)
            case b"\x0c":
                i = _tag_Long_Array(data, i, NBT_data, False)
            case _:
                raise Exception(f"Unknown Tag: {tag}")
        x = x + 1
