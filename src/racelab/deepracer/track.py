import pkg_resources
import os


def track(name=None):
    """
    Retrieve the file path of a track or list available track names.

    If the `name` is provided, returns the file path to the `.npy` track file. If the `name` is not found,
    a `FileNotFoundError` is raised.

    If `name` is `None`, lists all available track names under the 'deepracer/data' directory as comma-separated strings.

    Parameters
    ----------
    name : str, optional
        The name of the track to retrieve. If `None`, the function will list all available track names.

    Returns
    -------
    str
        The file path of the requested track if `name` is provided, or a comma-separated string of track names if `name` is `None`.

    Raises
    ------
    FileNotFoundError
        If `name` is provided but does not match any file in the `data` folder.
    """

    # Get the current module's name dynamically
    package_name = __name__.split(".")[
        0
    ]  # 'racelab' if the file is in 'racelab.deepracer'

    # Get the path to the 'data' folder in the current package (e.g., 'racelab.deepracer/data')
    data_folder = pkg_resources.resource_filename(package_name, "deepracer/data")

    if name is None:
        # List all .npy files in the 'data' folder and strip the '.npy' extension
        track_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(data_folder)
            if f.endswith(".npy")
        ]

        # Print available track names as a comma-separated string
        print("Available track names:", ", ".join(track_names))
    else:
        # Check if the requested track exists in the data folder
        if f"{name}.npy" not in os.listdir(data_folder):
            raise FileNotFoundError(f"Track '{name}' not found in the 'data' folder.")

        # Return the path for the specific track if a name is provided
        return pkg_resources.resource_filename(
            package_name, f"deepracer/data/{name}.npy"
        )
