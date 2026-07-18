__all__ = ["blocks_ignore", "version_data"]

import VersionDataProcesser

blocks_ignore = ["minecraft:air", "minecraft:cave_air"]
version_data = VersionDataProcesser.load_version_data("Minecraft_Version_Data.csv")

if False:
    VersionDataProcesser._process_raw_version_data(
        "Data Files/data.csv",
        "Minecraft_Version_Data.csv",
        [
            "Client version",
            "Protocol version",
            "Data version",
            "Resource pack format",
            "Data pack format",
        ],
    )
