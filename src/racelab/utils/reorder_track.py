import numpy as np
from shapely.geometry.polygon import LineString


def reorder_track(track):
    """
    Reorder the lines of a track array based on their directional distance
    to the first reference line in the track. The lines are sorted based
    on the length of each line, from the longest (outer bounds) to the shortest
    (inner bounds).

    Parameters
    ----------
    track : numpy.ndarray
        A 2D array where each row represents waypoints, and columns are
        arranged in pairs (x, y coordinates). The number of columns must be
        even, with each pair corresponding to one line in the track.

    Returns
    -------
    numpy.ndarray
        A reordered 2D array where the columns are rearranged such that the
        lines are sorted based on their length from outer bounds (longest)
        to inner bounds (shortest).

    Raises
    ------
    ValueError
        If the number of columns in the `track` is odd, indicating that the
        track does not consist of pairs of x, y coordinates.
    """
    if track.shape[1] % 2 != 0:
        raise ValueError(
            "Track must have an even number of columns (pairs of x, y coordinates)."
        )

    num_lines = track.shape[1] // 2
    line_data = []

    for i in range(num_lines):
        line_coords = track[:, 2 * i : 2 * i + 2]
        line_data.append(
            (LineString(line_coords).length, line_coords)
        )

    sorted_lines = sorted(line_data, key=lambda x: x[0], reverse=True)
    sorted_track = np.hstack([line[1] for line in sorted_lines])

    return sorted_track
