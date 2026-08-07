__all__ = ["blocks_ignore", "version_data"]

import VersionDataProcesser

blocks_ignore = ["minecraft:air", "minecraft:cave_air"]
version_data = VersionDataProcesser._load_version_data("Minecraft_Version_Data.csv")
