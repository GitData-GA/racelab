import numpy as np
import os
from racelab.utils.reorder_track import reorder_track


def load(path):
    """
    Load and reorder a track from a `.npy` file.

    This function loads a track stored in a `.npy` file, validates its structure,
    and reorders its lines (x, y coordinate pairs) based on their proximity to the
    overall centroid. The input file must represent the track as a 2D NumPy array
    where each row corresponds to a waypoint, and columns are organized in pairs
    (x, y coordinates) representing multiple lines.

    The function ensures that:
    - The file has the `.npy` extension.
    - The data is a 2D array with at least 2 waypoints.
    - The number of columns is even and sufficient to represent at least 3 lines.

    After validation, the function reorders the track lines to ensure they are sorted
    by their proximity to the centroid, with the innermost line first.

    Parameters
    ----------
    path : str
        The file path to the track file. The file must be a `.npy` file.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the reordered track. Each row corresponds to
        a waypoint, and each column pair represents (x, y) coordinates.
    """
    _, file_extension = os.path.splitext(path)

    if file_extension != ".npy":
        raise ValueError(
            f"A {file_extension} file detected. Please provide a '.npy' file."
        )

    track = np.load(path)

    if track.ndim != 2:
        raise ValueError(
            f"{track.ndim} dimension(s) detected. A track must be a 2D array."
        )

    if track.shape[0] < 2:
        raise ValueError(
            f"{track.shape[0]} waypoint detected. A tracks must have at least 2 waypoints."
        )

    if track.shape[1] % 2 != 0:
        raise ValueError(
            f"{track.shape[1]} columns detected. A track must have an even number of columns (e.g., x, y coordinates)."
        )

    if track.shape[1] < 6:
        raise ValueError(
            f"{track.shape[1]} columns detected. A track must have at least 6 columns, 3 pairs of (x, y) coordinates."
        )

    track = reorder_track(track)

    return track
