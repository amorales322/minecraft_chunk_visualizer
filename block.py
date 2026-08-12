__all__ = ["Block"]

from program_references import blocks_ignore


class Block:
    """
    Class representing a block and its properties.
    """

    def __init__(self, name: str, properties: dict[str, str | int]):
        self.namespace, self.name = name.split(":")
        self.properties = properties
        self.is_empty = True if name in blocks_ignore else False

    def __str__(self):
        return f"{self.namespace}:{self.name} -> {self.properties}"

    def to_NBT_format(self):
        formatted_data = {"Name": ":".join([self.namespace, self.name])}
        if self.properties:
            formatted_data["Properties"] = self.properties
        return formatted_data
