from struct import unpack
from typing import Any
import numpy as np

__all__ = [
    "NBTData",
    "UnknownTagError",
    "BYTE_SIGNED_DTYPE",
    "BYTE_UNSIGNED_DTYPE",
    "SHORT_SIGNED_DTYPE",
    "SHORT_UNSIGNED_DTYPE",
    "INT_SIGNED_DTYPE",
    "INT_UNSIGNED_DTYPE",
    "LONG_SIGNED_DTYPE",
]

BYTE_SIGNED_DTYPE = np.dtype(">b")
BYTE_UNSIGNED_DTYPE = np.dtype(">B")
SHORT_SIGNED_DTYPE = np.dtype(">h")
SHORT_UNSIGNED_DTYPE = np.dtype(">H")
INT_SIGNED_DTYPE = np.dtype(">i")
INT_UNSIGNED_DTYPE = np.dtype(">I")
LONG_SIGNED_DTYPE = np.dtype(">l")


class UnknownTagError(Exception):
    def __init__(self, tag_id: bytes):
        print(f"Unknown Tag ID: {tag_id}")


class NBTData:
    """
    NBTData class containing deserialized NBT Data.
    """

    def __init__(self, data, metadata):
        self.nbt_data = data
        self.nbt_structure = metadata

    @staticmethod
    def read(byte_data: bytes) -> "NBTData":
        """
        Returns generated NBT data deserialized from chunk byte data.

        :param byte_data: Raw chunk byte data.
        :type byte_data: bytes
        :return: 'NBTData' object containing the deserialized NBT data for the chunk.
        :rtype: NBTData
        :raise UnknownTagError: Raised if the tag ID is invalid for a specific tag payload pair. May indicate corrupted chunk data or data not properly decompressed.
        """
        NBT_data = {}
        NBT_metadata = {}
        tag_name, i = _extract_tag_name(byte_data, 0)
        raw_data_len = len(byte_data)
        while i < raw_data_len:
            tag = byte_data[i : i + 1]
            match tag:
                case b"\x00":
                    return NBTData(NBT_data, NBT_metadata)
                case b"\x01":
                    i = _tag_Byte(byte_data, NBT_data, NBT_metadata, i)
                case b"\x02":
                    i = _tag_Short(byte_data, NBT_data, NBT_metadata, i)
                case b"\x03":
                    i = _tag_Int(byte_data, NBT_data, NBT_metadata, i)
                case b"\x04":
                    i = _tag_Long(byte_data, NBT_data, NBT_metadata, i)
                case b"\x05":
                    i = _tag_Float(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x06":
                    i = _tag_Double(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x07":
                    i = _tag_Byte_Array(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x08":
                    i = _tag_String(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x09":
                    i = _tag_List(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x0a":
                    i = _tag_Compound(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x0b":
                    i = _tag_Int_Array(byte_data, NBT_data, NBT_metadata, i, False)
                case b"\x0c":
                    i = _tag_Long_Array(byte_data, NBT_data, NBT_metadata, i, False)
                case _:
                    raise UnknownTagError(tag)

        return NBTData(NBT_data, NBT_metadata)


def _extract_tag_name(byte_data: bytes, i: int) -> tuple[str | None, int]:
    name_len = int.from_bytes(byte_data[i + 1 : i + 3], "big")
    tag_name = byte_data[i + 3 : i + 3 + name_len].decode() if name_len != 0 else None
    return tag_name, i + 3 + name_len


def _tag_Byte(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
) -> int:
    name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    NBT_data[name] = int.from_bytes(byte_data[i : i + 1], "big", signed=True)
    NBT_metadata[name] = "byte"

    return i + 1


def _tag_Short(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
) -> int:
    name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    NBT_data[name] = int.from_bytes(byte_data[i : i + 2], "big", signed=True)
    NBT_metadata[name] = "short"
    return i + 2


def _tag_Int(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
) -> int:
    name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    NBT_data[name] = int.from_bytes(byte_data[i : i + 4], "big", signed=True)
    NBT_metadata[name] = "int"
    return i + 4


def _tag_Long(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
) -> int:
    name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    NBT_data[name] = int.from_bytes(byte_data[i : i + 8], "big", signed=True)
    NBT_metadata[name] = "long"
    return i + 8


def _tag_Float(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    if parent_list:
        NBT_data.append(unpack(">f", byte_data[i : i + 4]))
        NBT_metadata.append("float")
    else:
        NBT_data[name] = unpack(">f", byte_data[i : i + 4])
        NBT_metadata[name] = "float"
    return i + 4


def _tag_Double(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    if parent_list:
        NBT_data.append(unpack(">d", byte_data[i : i + 8]))
        NBT_metadata.append("double")
    else:
        NBT_data[name] = unpack(">d", byte_data[i : i + 8])
        NBT_metadata[name] = "double"
    return i + 8


def _tag_String(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name
    tag_data_length = int.from_bytes(byte_data[i : i + 2], "big")
    tag_data = byte_data[i + 2 : i + 2 + tag_data_length].decode()
    if parent_list:
        NBT_data.append(tag_data)
        NBT_metadata.append("string")
    else:
        NBT_data[name] = tag_data
        NBT_metadata[name] = "string"
    return i + 2 + tag_data_length


def _tag_Byte_Array(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name

    array_size = int.from_bytes(byte_data[i : i + 4], "big", signed=True)
    if parent_list:
        NBT_data.append(
            np.frombuffer(
                byte_data[i : i + 4 + array_size], dtype=BYTE_SIGNED_DTYPE, offset=4
            )
        )
        NBT_metadata.append("ByteArray")
    else:
        NBT_data[name] = np.frombuffer(
            byte_data[i : i + 4 + array_size], dtype=BYTE_SIGNED_DTYPE, offset=4
        )
        NBT_metadata[name] = "ByteArray"
    return i + 4 + array_size


def _tag_Int_Array(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name

    array_size = int.from_bytes(byte_data[i : i + 4], "big", signed=True)
    if parent_list:
        NBT_data.append(
            np.frombuffer(
                byte_data[i : i + 4 + (array_size * 4)],
                dtype=INT_SIGNED_DTYPE,
                offset=4,
            )
        )
        NBT_metadata.append("IntArray")
    else:
        NBT_data[name] = np.frombuffer(
            byte_data[i : i + 4 + (array_size * 4)], dtype=INT_SIGNED_DTYPE, offset=4
        )
        NBT_metadata[name] = "IntArray"
    return i + 4 + (array_size * 4)


def _tag_Long_Array(
    byte_data: bytes,
    NBT_data: dict[str, Any] | list[Any],
    NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name

    array_size = int.from_bytes(byte_data[i : i + 4], "big", signed=True)
    if parent_list:
        NBT_data.append(
            np.frombuffer(
                byte_data[i : i + 4 + (array_size * 8)],
                dtype=LONG_SIGNED_DTYPE,
                offset=4,
            )
        )
        NBT_metadata.append("LongArray")
    else:
        NBT_data[name] = np.frombuffer(
            byte_data[i : i + 4 + (array_size * 8)], dtype=LONG_SIGNED_DTYPE, offset=4
        )
        NBT_metadata[name] = "LongArray"
    return i + 4 + (array_size * 8)


def _tag_List(
    byte_data: bytes,
    root_NBT_data: dict[str, Any] | list[Any],
    root_NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    NBT_data = []
    NBT_metadata = []
    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name

    tag_ID = byte_data[i : i + 1]
    array_size = int.from_bytes(byte_data[i + 1 : i + 5], "big", signed=True)
    i = i + 1 + 4

    if array_size == 0:
        if parent_list:
            root_NBT_data.append([])
            root_NBT_metadata.append([])
        else:
            root_NBT_data[name] = []
            root_NBT_metadata[name] = []
        return i

    match tag_ID:
        case b"\x00":
            NBT_metadata.append("End")
        case b"\x01":
            NBT_metadata.append("byte")
        case b"\x02":
            NBT_metadata.append("short")
        case b"\x03":
            NBT_metadata.append("int")
        case b"\x04":
            NBT_metadata.append("long")
        case b"\x05":
            NBT_metadata.append("float")
        case b"\x06":
            NBT_metadata.append("double")
        case b"\x07":
            NBT_metadata.append("ByteArray")
        case b"\x08":
            NBT_metadata.append("string")
        case b"\x09":
            NBT_metadata.append("List")
        case b"\x0a":
            NBT_metadata.append("Compound")
        case b"\x0b":
            NBT_metadata.append("IntArray")
        case b"\x0c":
            NBT_metadata.append("LongArray")
        case _:
            raise UnknownTagError(tag_ID)

    for x in range(0, array_size, 1):
        match tag_ID:
            case b"\x00":
                if parent_list:
                    root_NBT_data.append(NBT_data)
                    root_NBT_metadata.append(NBT_metadata)
                else:
                    root_NBT_data[name] = NBT_data
                    root_NBT_metadata[name] = NBT_metadata
                return i + 1
            case b"\x01":
                NBT_data.append(
                    np.frombuffer(
                        byte_data[i : i + array_size],
                        dtype=BYTE_SIGNED_DTYPE,
                    )
                )
                i += array_size
                break
            case b"\x02":
                NBT_data.append(
                    np.frombuffer(
                        byte_data[i : i + (array_size * 2)],
                        dtype=SHORT_SIGNED_DTYPE,
                    )
                )
                i += array_size * 2
                break
            case b"\x03":
                NBT_data.append(
                    np.frombuffer(
                        byte_data[i : i + (array_size * 4)],
                        dtype=INT_SIGNED_DTYPE,
                    )
                )
                i += array_size * 4
                break
            case b"\x04":
                NBT_data.append(
                    np.frombuffer(
                        byte_data[i : i + (array_size * 8)],
                        dtype=LONG_SIGNED_DTYPE,
                    )
                )
                i += array_size * 8
                break
            case b"\x05":
                i = _tag_Float(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x06":
                i = _tag_Double(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x07":
                i = _tag_Byte_Array(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x08":
                i = _tag_String(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x09":
                i = _tag_List(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x0a":
                i = _tag_Compound(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x0b":
                i = _tag_Int_Array(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case b"\x0c":
                i = _tag_Long_Array(byte_data, NBT_data, NBT_metadata, i, True, str(x))
            case _:
                raise UnknownTagError(tag_ID)

    if parent_list:
        root_NBT_data.append(NBT_data)
        root_NBT_metadata.append(NBT_metadata)
    else:
        root_NBT_data[name] = NBT_data
        root_NBT_metadata[name] = NBT_metadata
    return i


def _tag_Compound(
    byte_data: bytes,
    root_NBT_data: dict[str, Any] | list[Any],
    root_NBT_metadata: dict[str, Any] | list[Any],
    i: int,
    parent_list: bool,
    name: str | None = None,
) -> int:
    NBT_data = {}
    NBT_metadata = {}

    if not parent_list:
        name, i = _extract_tag_name(byte_data, i)
    name = str(i) if name is None else name

    x = 0
    while True:
        tag = byte_data[i : i + 1]
        match tag:
            case b"\x00":
                if parent_list:
                    root_NBT_data.append(NBT_data)
                    root_NBT_metadata.append(NBT_metadata)
                else:
                    root_NBT_data[name] = NBT_data
                    root_NBT_metadata[name] = NBT_metadata
                return i + 1
            case b"\x01":
                i = _tag_Byte(byte_data, NBT_data, NBT_metadata, i)
            case b"\x02":
                i = _tag_Short(byte_data, NBT_data, NBT_metadata, i)
            case b"\x03":
                i = _tag_Int(byte_data, NBT_data, NBT_metadata, i)
            case b"\x04":
                i = _tag_Long(byte_data, NBT_data, NBT_metadata, i)
            case b"\x05":
                i = _tag_Float(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x06":
                i = _tag_Double(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x07":
                i = _tag_Byte_Array(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x08":
                i = _tag_String(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x09":
                i = _tag_List(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x0a":
                i = _tag_Compound(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x0b":
                i = _tag_Int_Array(byte_data, NBT_data, NBT_metadata, i, False)
            case b"\x0c":
                i = _tag_Long_Array(byte_data, NBT_data, NBT_metadata, i, False)
            case _:
                raise UnknownTagError(tag)
        x = x + 1
