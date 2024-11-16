from shapely.geometry import Point, Polygon
from racelab.optimizer.utils.k1999 import menger_curvature, refine_point, refine_line


def k1999(track, line_iterations, xi_iterations, margin=0, atol=1e-3):
    """
    Implements the K1999 algorithm to compute the optimal racing line for a given track.

    Parameters
    ----------
    track : numpy.ndarray
        A 2D array where each row represents waypoints on the track. Columns are 
        structured in pairs as (x, y) coordinates, with the outer border first 
        and the inner border last.
    line_iterations : int
        The number of iterations to refine the racing line for optimality.
    xi_iterations : int
        The number of iterations to refine individual waypoints along the racing line.
    margin : float, optional
        A scaling factor to offset the inner and outer borders towards the middle of the track.
        Positive values pull the edges inward, creating a buffer zone that prevents the
        optimal line from hugging the track boundaries too closely. Defaults to 0 (no margin).
    atol : float, optional
        The absolute tolerance for convergence during refinement. Defaults to 1e-3.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the optimized racing line, with each row as (x, y) coordinates.

    Notes
    -----
    - The algorithm starts with the middle line calculated as the average of the inner 
      and outer borders.
    - Margins are applied by adjusting the inner and outer borders towards the middle line.
    - The refinement process ensures the racing line lies within the adjusted borders 
      while optimizing for smoother curvature.
    """
    if margin < 0:
        raise ValueError(
            f"A {margin} margin detected. The margin must be non-negative."
        )

    num_columns = track.shape[1]
    outer_border = track[:, 0:2]
    inner_border = track[:, num_columns - 2 : num_columns]
    refined_line = (outer_border + inner_border) / 2.0

    inner_polygon = Polygon(inner_border + margin * (refined_line - inner_border))
    outer_polygon = Polygon(outer_border - margin * (outer_border - refined_line))

    # Perform the line iterations
    for _ in range(line_iterations):
        refined_line = refine_line(
            track, refined_line, inner_polygon, outer_polygon, xi_iterations, atol
        )

    return refined_line
