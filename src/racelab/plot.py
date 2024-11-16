import matplotlib.pyplot as plt
import numpy as np


def plot(track, figsize=(10, 6), savefig_options=None):
    """
    Draw and optionally save the track.

    Parameters
    ----------
    track : numpy.ndarray
        A 2D array where each row represents waypoints, with columns arranged
        in pairs (x, y coordinates). Columns are ordered sequentially to represent
        the inner, middle, and outer racing lines.
    figsize : tuple, optional
        The size of the figure, by default (10, 6).
    savefig_options : dict, optional
        A dictionary of options for saving the plot. If provided, it should contain:
        - 'fname' (str, optional): The file path for saving the image.
        - Additional keyword arguments supported by `matplotlib.pyplot.savefig`.

    Returns
    -------
    None
        Displays the racing line plot and optionally saves it to a file if
        `savefig_options` is provided.
    """
    num_lines = track.shape[1] // 2
    plt.figure(figsize=figsize)

    for i in range(num_lines):
        x_coords = track[:, 2 * i]
        y_coords = track[:, 2 * i + 1]
        plt.plot(x_coords, y_coords)

    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)

    if savefig_options:
        savefig_kwargs = {k: v for k, v in savefig_options.items()}
        plt.savefig(**savefig_kwargs)

    plt.show()
