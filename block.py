__all__ = ["Block"]


class Block:
    """
    Class representing a block and its properties.
    """

    def __init__(self, name: str, properties: dict[str, str | int]):
        self.namespace, self.name = name.split(":")
        self.properties = properties

    def __str__(self):
        return f'{self.namespace}:{self.name} -> {self.properties}'

    def _to_NBT_format(self):
        formatted_data = {"Name": ":".join([self.namespace, self.name])}
        if self.properties:
            formatted_data["Properties"] = self.properties
        return formatted_data
