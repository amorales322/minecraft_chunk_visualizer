import NBTParser
import ChunkParser
import Plotter
import _GlobalReferences

import itertools

region_data = NBTParser.read_region_file("TestFiles/r.0.0.mca")
for x, z in itertools.product(range(0, 1), range(0, 1)):
    chunk_data = NBTParser.extract_chunk_data(region_data, x, z)
    extracted_NBT_data = NBTParser.generate_NBT_Data(chunk_data)
    ChunkParser.parse_chunk(
        extracted_NBT_data,
        extracted_NBT_data["xPos"],
        extracted_NBT_data["zPos"],
        _GlobalReferences.point_data,
        _GlobalReferences.face_data,
        _GlobalReferences.block_data,
    )
Plotter.plot(_GlobalReferences.point_data, _GlobalReferences.face_data)
