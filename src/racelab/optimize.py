import copy
from racelab.optimizer.k1999 import k1999


def optimize(track, optimizer, **param):
    """
    Optimize the racing line using a specified optimization algorithm.

    Parameters
    ----------
    track : numpy.ndarray
        A 2D array representing the racetrack, where each row corresponds to a set
        of waypoints, and columns represent different lines (e.g., outer, middle, inner).
    optimizer : str
        The name of the optimization algorithm to use. Must be one of the supported
        optimizers listed in the function (e.g., 'k1999', 'custom').
    **param
        Additional arguments required by the selected optimizer. For the 'custom'
        optimizer, a 'custom_func' argument must be provided as the custom optimization
        function.

    Returns
    -------
    numpy.ndarray
        The optimized racing line as a 2D array of waypoints.

    Raises
    ------
    ValueError
        If an unsupported optimizer is specified, or if 'custom' is selected but
        no valid 'custom_func' is provided.
    """
    track_copy = copy.deepcopy(track)

    optimizer_dispatch = {
        "k1999": k1999,
        "custom": None,
    }

    optimizer_func = optimizer_dispatch.get(optimizer)

    if optimizer == "custom":
        custom_func = param.pop("custom_func", None)
        if not callable(custom_func):
            raise ValueError(
                "For the 'custom' optimizer, a 'custom_func' argument must be provided, and it must be callable."
            )
        optimizer_func = custom_func

    if not optimizer_func:
        raise ValueError(
            f"Unsupported optimizer: '{optimizer}'. Supported optimizers are: {', '.join(optimizer_dispatch.keys())}."
        )

    optimal_line = optimizer_func(track_copy, **param)

    return optimal_line
