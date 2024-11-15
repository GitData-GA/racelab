import numpy as np


def reorder_track(track):
    """
    Reorder the lines of a track array based on proximity to the centroid.

    This function takes a 2D NumPy array representing a track with multiple
    lines (x, y coordinate pairs) and reorders the lines by their geometric
    proximity to the overall centroid. The centroid is calculated as the
    mean of all coordinates, and lines closer to it are ordered first.

    Parameters
    ----------
    track : numpy.ndarray
        A 2D array where each row represents waypoints, and columns are
        arranged in pairs (x, y coordinates). The number of columns must
        be even, with each pair corresponding to one line.

    Returns
    -------
    numpy.ndarray
        A reordered 2D array where the columns are rearranged such that
        the lines (x, y pairs) are sorted by their distance to the overall
        centroid in ascending order.
    """
    overall_centroid = np.mean(track, axis=(0, 1))

    num_lines = track.shape[1] // 2
    line_data = []

    for i in range(num_lines):
        line_coords = track[:, 2 * i : 2 * i + 2]
        line_centroid = np.mean(line_coords, axis=0)
        distance = np.linalg.norm(line_centroid - overall_centroid)
        line_data.append((distance, line_coords))

    sorted_lines = sorted(line_data, key=lambda x: x[0])

    return np.hstack([line[1] for line in sorted_lines])
