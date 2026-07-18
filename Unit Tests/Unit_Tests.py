import NBTParser
import ChunkParser
import Plotter

import pytest
from os import listdir
from itertools import product as iter_product


@pytest.mark.parametrize("file", listdir("Data Files/MCA Testing Files"))
def test_region_parsing(file):
    assert NBTParser.Region.from_region_file(
        f"DataFiles/MCA Testing Files/{file}", True
    )


@pytest.mark.parametrize("file", listdir("Data Files/MCA Testing Files"))
def test_plot(file):
    points = []
    faces = []
    blocks = []
    region_data = NBTParser.Region.from_region_file(
        f"Data Files/MCA Testing Files/{file}", True
    )
    for x, z in iter_product(range(0, 8), range(0, 8)):
        chunk_data = NBTParser.Chunk.from_region_data(region_data, x, z)
        ChunkParser.parse_chunk(
            chunk_data.data,
            points,
            faces,
            blocks,
        )
    Plotter.plot(points, faces, True)
