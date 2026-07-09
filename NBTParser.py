__all__ = (
    [
        "read_region_file",
        "export_to_JSON",
        "extract_chunk_data",
        "generate_NBT_Data",
    ],
)

from struct import unpack
from typing import Any
from zlib import decompress
import json


def read_region_file(filepath: str) -> bytes:
    with open(filepath, "rb") as file:
        data = file.read()
    return data

def export_to_JSON(data: dict[str, Any], path: str) -> None:
    with open(path, "w") as file:
        json.dump(data, file)


def extract_chunk_data(data, xpos, zpos):
    header_offset = 4 * ((xpos % 32) + (zpos % 32) * 32)

    # Assumed to be unsigned, it is not stated if this value is signed or unsigned
    chunk_sector_offset = (
        int.from_bytes(data[header_offset : header_offset + 3], "big") * 4096
    )
    chunk_sector_count = int.from_bytes(
        data[header_offset + 3 : header_offset + 4], "big"
    )

    chunk_byte_size = data[chunk_sector_offset : chunk_sector_offset + 4]

    # To-Do: Throw error with unknown compression type
    compression_type = data[chunk_sector_offset + 4 : chunk_sector_offset + 5]

    return decompress(
        data[
            chunk_sector_offset + 5 : chunk_sector_offset
            + 5
            + int.from_bytes(chunk_byte_size, "big", signed=True)
        ]
    )


def generate_NBT_Data(data: bytes) -> dict[str, Any]:
    NBT_data = {}
    tag_name, tag_name_length, i = _extract_tag_name(data, 0)

    while i < len(data):
        tag = data[i : i + 1]
        match tag:
            case b"\x00":
                return NBT_data
            case b"\x01":
                i = _tag_Byte(data, i, NBT_data)
            case b"\x02":
                i = _tag_Short(data, i, NBT_data)
            case b"\x03":
                i = _tag_Int(data, i, NBT_data)
            case b"\x04":
                i = _tag_Long(data, i, NBT_data)
            case b"\x05":
                i = _tag_Float(data, i, NBT_data)
            case b"\x06":
                i = _tag_Double(data, i, NBT_data)
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
                raise Exception("Unknown Tag")

    return NBT_data


def _extract_tag_name(data: bytes, i: int) -> tuple[str, int, int]:
    tag_name_length = int.from_bytes(data[i + 1 : i + 3], "big")
    tag_name = (
        data[i + 3 : i + 3 + tag_name_length].decode()
        if tag_name_length != 0
        else str(i)
    )

    return tag_name, tag_name_length, i + 3 + tag_name_length


def _tag_Byte(data: bytes, i: int, NBT_data: dict[str, Any]) -> int:
    tag_name, tag_name_length, i = _extract_tag_name(data, i)
    tag_data = int.from_bytes(data[i : i + 1], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 1


def _tag_Short(data: bytes, i: int, NBT_data: dict[str, Any]) -> int:
    tag_name, tag_name_length, i = _extract_tag_name(data, i)
    tag_data = int.from_bytes(data[i : i + 2], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 2


def _tag_Int(data: bytes, i: int, NBT_data: dict[str, Any]) -> int:
    tag_name, tag_name_length, i = _extract_tag_name(data, i)
    tag_data = int.from_bytes(data[i : i + 4], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 4


def _tag_Long(data: bytes, i: int, NBT_data: dict[str, Any]) -> int:
    tag_name, tag_name_length, i = _extract_tag_name(data, i)
    tag_data = int.from_bytes(data[i : i + 8], "big", signed=True)
    NBT_data[tag_name] = tag_data
    return i + 8


def _tag_Float(data: bytes, i: int, NBT_data: dict[str, Any]) -> int:
    tag_name, tag_name_length, i = _extract_tag_name(data, i)
    tag_data = unpack(">f", data[i : i + 4])
    NBT_data[tag_name] = tag_data
    return i + 4


def _tag_Double(data: bytes, i: int, NBT_data: dict[str, Any]) -> int:
    tag_name, tag_name_length, i = _extract_tag_name(data, i)
    tag_data = unpack(">d", data[i : i + 8])
    NBT_data[tag_name] = tag_data
    return i + 8


def _tag_String(
    data: bytes, i: int, NBT_data: dict[str, Any], root_is_list: bool
) -> int:
    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i)

    tag_data_length = int.from_bytes(data[i : i + 2], "big")
    tag_data = data[i + 2 : i + 2 + tag_data_length].decode()
    NBT_data[tag_name] = tag_data

    return i + 2 + tag_data_length


def _tag_Byte_Array(
    data: bytes, i: int, NBT_data: dict[str, Any], root_is_list: bool
) -> int:
    byte_array = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i)

    array_size = int.from_bytes(data[i : i + 4], "big", signed=True)
    i = i + 4

    for x in range(0, array_size, 1):
        tag_data = int.from_bytes(data[i : i + 1], "big", signed=True)
        byte_array[str(x)] = tag_data
        i = i + 1

    NBT_data[tag_name] = byte_array
    return i


def _tag_Int_Array(
    data: bytes, i: int, NBT_data: dict[str, Any], root_is_list: bool
) -> int:
    int_array = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i)

    array_size = int.from_bytes(data[i : i + 4], "big", signed=True)
    i = i + 4

    for x in range(0, array_size, 1):
        tag_data = int.from_bytes(data[i : i + 4], "big", signed=True)
        int_array[str(x)] = tag_data
        i = i + 4

    NBT_data[tag_name] = int_array
    return i


def _tag_Long_Array(data: bytes, i: int, NBT_data: dict[str, Any], root_is_list) -> int:
    long_array = {}

    if not root_is_list:
        tag_name, tag_name_length, i = _extract_tag_name(data, i)
    else:
        tag_name = str(i)

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
                i = _tag_Byte(data, i, NBT_data)
            case b"\x02":
                i = _tag_Short(data, i, NBT_data)
            case b"\x03":
                i = _tag_Int(data, i, NBT_data)
            case b"\x04":
                i = _tag_Long(data, i, NBT_data)
            case b"\x05":
                i = _tag_Float(data, i, NBT_data)
            case b"\x06":
                i = _tag_Double(data, i, NBT_data)
            case b"\x07":
                i = _tag_Byte_Array(data, i, NBT_data, True)
            case b"\x08":
                i = _tag_String(data, i, NBT_data, True)
            case b"\x09":
                i = _tag_List(data, i, NBT_data, True, str(x))
            case b"\x0a":
                i = _tag_Compound(data, i, NBT_data, True, str(x))
            case b"\x0b":
                i = _tag_Int_Array(data, i, NBT_data, True)
            case b"\x0c":
                i = _tag_Long_Array(data, i, NBT_data, True)
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
                i = _tag_Byte(data, i, NBT_data)
            case b"\x02":
                i = _tag_Short(data, i, NBT_data)
            case b"\x03":
                i = _tag_Int(data, i, NBT_data)
            case b"\x04":
                i = _tag_Long(data, i, NBT_data)
            case b"\x05":
                i = _tag_Float(data, i, NBT_data)
            case b"\x06":
                i = _tag_Double(data, i, NBT_data)
            case b"\x07":
                i = _tag_Byte_Array(data, i, NBT_data, False)
            case b"\x08":
                i = _tag_String(data, i, NBT_data, False)
            case b"\x09":
                i = _tag_List(data, i, NBT_data, False, str(x))
            case b"\x0a":
                i = _tag_Compound(data, i, NBT_data, False, str(x))
            case b"\x0b":
                i = _tag_Int_Array(data, i, NBT_data, False)
            case b"\x0c":
                i = _tag_Long_Array(data, i, NBT_data, False)
            case _:
                raise Exception("Unknown Tag")
        x = x + 1
