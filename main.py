import NBTParser
import ChunkParser
import Plotter
import Profiler

from itertools import product as iter_product


def profiler_func(**kwargs):
    for x, z in iter_product(range(0, 4), range(0, 4)):
        chunk_data = NBTParser.Chunk.from_region_data(kwargs["region_data"], x, z)
        ChunkParser.parse_chunk(
            chunk_data.data,
            point_data,
            face_data,
            block_data,
        )


block_data = []
point_data = []
face_data = []

region_data = NBTParser.Region.from_region_file(
    "Data Files/MCA Testing Files/r.-2.2-26.1.1.mca", True
)
Profiler.run_profiler(profiler_func, region_data=region_data)
Plotter.plot(point_data, face_data)
