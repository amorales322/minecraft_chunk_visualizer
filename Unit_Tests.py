import Region
import Chunk

import pytest
from os import listdir
from itertools import product as iter_product


@pytest.mark.parametrize("file", listdir("Data Files/MCA Testing Files"))
def test_region_parsing(file):
    assert Region.Region.from_region_file(
        f"Data Files/MCA Testing Files/{file}", True
    )


@pytest.mark.parametrize("file", listdir("Data Files/MCA Testing Files"))
def test_plot(file):
    mesh_array = []
    region_data = Region.Region.from_region_file(
        f"Data Files/MCA Testing Files/{file}", True
    )
    for x, z in iter_product(range(0, 8), range(0, 8)):
        chunk_data = Chunk.Chunk.from_region_data(region_data, x, z)
        mesh_array.extend(chunk_data.generate_mesh())
