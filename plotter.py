__all__ = ["plot"]

import pyvista as pv

from mesh import Mesh


def plot(
    mesh_data: list[Mesh],
    /,
    *,
    wireframe: bool = False,
    window_size: tuple[int, int] = (1920, 1080),
    camera_position: str = "zx",
) -> None:
    """
    Plots the mesh data using the pyvista Plotter class.

    :param mesh_data: List of pyvista PolyData objects to plot.
    :type mesh_data: list[pyvista.PolyData]
    :param wireframe: Enable/Disable wireframe.
    :type wireframe: bool
    :param window_size: Window size.
    :type window_size: tuple[int, int]
    :param camera_position: Initial camera axis position.
    :type camera_position: str
    :return: `None`
    :rtype: None
    """
    pl = pv.Plotter()
    if mesh_data:
        for mesh in mesh_data:
            if not mesh.is_empty:
                pl.add_mesh(mesh.mesh_data, show_edges=wireframe)
        pl.camera_position = camera_position
        pl.window_size = window_size
        pl.add_axes(
            interactive=True,
            line_width=5,
        )
        pl.show()
