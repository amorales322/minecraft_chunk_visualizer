__all__ = ["Coordinate"]


class Coordinate:
    """
    Coordinate class defining a coordinate.
    """

    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def get_block_index(self) -> int:
        """
        Calculates the section index of the block.

        :return: The index of the block based on the relative coordinates of a subchunk.
        :rtype: int
        """

        return self.get_section_relative_y() * 256 + (self.z % 16) * 16 + (self.x % 16)

    def get_section(self) -> int:
        """
        Calculates the subchunk number of the block.

        :return: The subchunk number based on the y-coordinate.
        :rtype: int
        """
        next_section = abs(self.y) // 16 + 1
        return (self.y + next_section * 16) // 16 - next_section

    def get_section_relative_y(self) -> int:
        """
        Calculates the y-coordinate of the block relative to the subchunk.

        :return: The relative subchunk y-coordinate of the block.
        :rtype: int
        """
        return (self.y + (abs(self.y) // 16 + 1) * 16) % 16
