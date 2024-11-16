import numpy as np

def reorder_track(track):
    """
    Reorder the lines of a track array based on their directional distance
    to the first reference line in the track. The lines are sorted based
    on the average distance (positive for outer bounds, negative for inner bounds).

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
        the lines are sorted from outer to inner based on their average distance.
    """
    # Ensure we have an even number of columns (pairs of x, y coordinates)
    if track.shape[1] % 2 != 0:
        raise ValueError("Track must have an even number of columns (pairs of x, y coordinates).")

    # Extract the reference line (first pair of x, y coordinates)
    reference_line = track[:, :2]  # First line (first two columns)
    
    num_lines = track.shape[1] // 2  # Number of lines in the track
    line_data = []

    for i in range(1, num_lines):  # Iterate over each line
        # Extract coordinates for the current line (x, y pairs)
        line_coords = track[:, 2 * i : 2 * i + 2]
        
        # Calculate Euclidean distance between each point in the current line and the reference line
        # We subtract the reference line's points from each of the line's points
        distances = np.linalg.norm(line_coords - reference_line, axis=1)  # Euclidean distance

        # We want to adjust the direction such that the reference line has distance 0
        # The reference line should have 0 distance from itself, so subtract the first point's distance
        distances -= distances[0]

        # Average the distances for this line
        avg_distance = np.mean(distances)
        line_data.append((avg_distance, line_coords))

    # Sort the lines based on the average distance, from outer to inner (descending)
    sorted_lines = sorted(line_data, key=lambda x: x[0], reverse=True)

    # Combine all lines (including the reference line) in the sorted order
    # Add the reference line at the beginning of the sorted lines
    sorted_track = np.hstack([reference_line] + [line[1] for line in sorted_lines])

    return sorted_track
