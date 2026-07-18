from typing import Any
import numpy as np

import Plotter


def parse_chunk(data: Any, points, faces, block_data):
    if data:
        for section_y in range(0, len(data["sections"])):
            palette = data["sections"][str(section_y)]["block_states"]["palette"]

            if len(palette) != 1:
                block_state_data = data["sections"][str(section_y)]["block_states"][
                    "data"
                ]
                block_buffer = []
                for i in range(0, len(block_state_data)):
                    bit_width = max(len(f"{len(palette) - 1:b}"), 4)
                    bin_val = np.binary_repr(block_state_data[str(i)], width=64)
                    for v in range(64, 64 % bit_width, -bit_width):
                        block_buffer.append(
                            palette[str(int(bin_val[v - bit_width : v], 2))]["Name"]
                        )
                block_data.append(block_buffer)

                Plotter.gen_section_mesh(
                    section_origin=(
                        16 * data["xPos"],
                        16 * section_y,
                        16 * data["zPos"],
                    ),
                    block_data=block_data[section_y],
                    points=points,
                    faces=faces,
                )
            else:
                block_name = palette["0"]["Name"]
                block_data.append([block_name] * 4096)
                Plotter.gen_section_mesh(
                    section_origin=(
                        16 * data["xPos"],
                        16 * section_y,
                        16 * data["zPos"],
                    ),
                    block_data=block_data[section_y],
                    points=points,
                    faces=faces,
                )
    else:
        print("Chunk is Empty.")
