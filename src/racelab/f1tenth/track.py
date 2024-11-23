import pkg_resources
import os


def track(name=None):
    """
    Retrieve the file path of a track or list available track names.

    If the `name` is provided, returns the file path to the `.npy` track file. If the `name` is not found,
    a `FileNotFoundError` is raised.

    If `name` is `None`, lists all available track names under the 'f1tenth/data' directory as comma-separated strings.

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

    package_name = __name__.split(".")[0]

    data_folder = pkg_resources.resource_filename(package_name, "f1tenth/data")

    track_names = [
        os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith(".csv")
    ]

    if name is None:
        print("Available track names:\n", '\n'.join(sorted(track_names)))
    else:
        if f"{name}.csv" not in os.listdir(data_folder):
            raise FileNotFoundError(
                f"Track '{name}' not found."
            )

        return pkg_resources.resource_filename(
            package_name, f"f1tenth/data/{name}.csv"
        )
