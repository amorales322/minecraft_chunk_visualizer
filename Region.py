from itertools import product as iter_product

from Chunk import Chunk

__all__ = ["Region"]


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
