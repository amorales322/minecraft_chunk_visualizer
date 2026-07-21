import Plotter
from Chunk import Chunk
from Region import Region

from itertools import product as iter_product

mesh_array = []

region_data = Region.from_region_file("Data Files/MCA Testing Files/r.0.0.mca", True)
for x, z in iter_product(range(0, 32), range(0, 32)):
    chunk_data = Chunk.from_region_data(region_data, x, z)
    mesh_array.extend(chunk_data.generate_mesh())
Plotter.plot(mesh_array, wireframe=False)
