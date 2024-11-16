import numpy as np
import os


def load_track(path):
    """Load a track from a file

    This function is designed to load and validate a track stored in a `.npy`
    file. A track is represented as a 2D NumPy array where each row corresponds
    to a waypoint, and each pair of columns represent (x, y) coordinates. The
    function ensures the file is correctly formatted and adheres to the expected
    structure.

    Parameters
    ----------
    path : str
        The file path to the track file. The file must be a `.npy` file.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the track. Each row corresponds to a waypoint,
        and each column pair represents (x, y) coordinates.
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

    if track.shape[1] % 2 != 0:
        raise ValueError(
            f"{track.shape[1]} columns detected. A track must have an even number of columns (e.g., x, y coordinates)."
        )

    if track.shape[0] < 2:
        raise ValueError(
            f"{track.shape[0]} waypoint detected. A tracks must have at least 2 waypoints."
        )

    return track
