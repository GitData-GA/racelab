import numpy as np

def reorder_track(track):
    """
    Reorder the lines of a track array based on their directional distance
    to the first reference line in the track. The lines are sorted based
    on the average distance (with direction indicated by positive or negative
    values) from every point of the reference line.

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
        the lines are sorted by their average directional distance to the
        first reference line.
    """
    # Extract the first pair of coordinates as the reference line
    reference_line = track[:, :2]  # First line (first two columns)
    
    num_lines = track.shape[1] // 2  # Number of lines in the track
    line_data = []

    for i in range(1, num_lines):  # Start from the second line to compare with the reference line
        line_coords = track[:, 2 * i : 2 * i + 2]
        
        # Calculate the directional distances for this line to the reference line
        distances = np.linalg.norm(line_coords - reference_line, axis=1)  # Distance between each point of the line and the reference line
        
        # Average the distances for this line
        avg_distance = np.mean(distances)
        line_data.append((avg_distance, line_coords))

    # Sort the lines based on the average distance, from outer to inner (descending)
    sorted_lines = sorted(line_data, key=lambda x: x[0], reverse=True)

    # Return the sorted lines as a single array
    return np.hstack([line[1] for line in sorted_lines])
