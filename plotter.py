__all__ = ["plot"]

import pyvista as pv

from mesh import Mesh


def plot(
    mesh_data: list[Mesh], wireframe: bool = False, unit_test_mode: bool = False
) -> None:
    """
    Plots the mesh data using the pyvista Plotter class.

    :param mesh_data: List of pyvista PolyData objects to plot
    :type mesh_data: list[pyvista.PolyData]
    :param wireframe: Enable/Disable wireframe.
    :type wireframe: bool
    :param unit_test_mode: Used for unit testing. Verifies the function works and skips plotting.
    :type unit_test_mode: bool
    :return: `None`
    :rtype: None
    """
    pl = pv.Plotter()
    if len(mesh_data) != 0:
        for mesh in mesh_data:
            if not mesh.is_empty:
                pl.add_mesh(mesh.mesh_data, show_edges=wireframe)

    else:
        print("Mesh is empty")
    if not unit_test_mode:
        pl.camera_position = "zx"
        pl.window_size = [1920, 1080]
        pl.add_axes(
            interactive=True,
            line_width=5,
        )
        pl.show()
