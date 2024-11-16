from shapely.geometry import Point, Polygon
from racelab.optimizer.utils.k1999 import menger_curvature, refine_point, refine_line


def k1999(track, line_iterations, xi_iterations, atol=1e-3):
    """
    K1999 algorithm to find the optimal racing line.

    Parameters
    ----------
    track : numpy.ndarray
        A 2D array where each row represents waypoints, and columns are arranged
        in pairs (x, y coordinates).
    line_iterations : int
        Number of iterations for refining the racing line.
    xi_iterations : int
        Number of iterations for refining each waypoint.

    Returns
    -------
    numpy.ndarray
        The optimized racing line as a 2D array.
    """
    num_columns = track.shape[1]
    outer_border = track[:, 0:2]
    inner_border = track[:, num_columns - 2 : num_columns]

    # Middle line
    middle_start = (num_columns // 2) - 1
    middle_end = middle_start + 2
    refined_line = track[:, middle_start:middle_end]

    inner_polygon = Polygon(inner_border)
    outer_polygon = Polygon(outer_border)

    # Perform the line iterations
    for _ in range(line_iterations):
        refined_line = refine_line(
            track, refined_line, inner_polygon, outer_polygon, xi_iterations, atol
        )

    return refined_line
