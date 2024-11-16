import numpy as np

def reorder_track(track):
    """
    Reorder the lines of a track array based on their distance from the first point of the track.
    The lines are sorted from the outer bound (farthest from the first point) 
    to the inner bound (closest to the first point).

    This function takes a 2D NumPy array representing a track with multiple
    lines (x, y coordinate pairs) and reorders the lines based on their distance
    from the first point in the track.

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
        the lines (x, y pairs) are sorted by their distance from the first point
        in descending order (farthest to closest).
    """
    # The first point in the track as the reference point
    reference_point = track[0]

    num_lines = track.shape[1] // 2
    line_data = []

    for i in range(num_lines):
        # Extract each line's coordinates
        line_coords = track[:, 2 * i : 2 * i + 2]
        # Find the distance from the first point to the centroid of this line
        line_centroid = np.mean(line_coords, axis=0)
        distance = np.linalg.norm(line_centroid - reference_point)
        line_data.append((distance, line_coords))

    # Sort the lines by distance in descending order (farthest to closest)
    sorted_lines = sorted(line_data, key=lambda x: x[0], reverse=True)

    return np.hstack([line[1] for line in sorted_lines])
