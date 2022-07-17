import numpy as np


def cont_shuffle_to_tbplw(origin: np.ndarray) -> np.ndarray:
    """shuffle matrix in CONT model to the order implemented in TBPLW

    Note: matrix can be hamk and bands[i]. The order now in CONT is:
    layer index, g index, A/B index. The order now in TBPLW is:
    layer index, A/B index, g index.

    ATTENTION: you need to make sure the order of Glist is consistent in both methods.

    Args:
        origin (np.ndarray): original matrix in the order of layer index, g index, A/B index.

    Returns:
        np.ndarray: output matrix in the order of layer index, A/B index, g index.
    """

    size = origin.shape[0]
    num_g = size//4
    origin_expanded = origin.reshape(2, num_g, 2, 2, num_g, 2)
    origin_shuffled = np.moveaxis(origin_expanded, [1, 4], [2, 5])

    return origin_shuffled.reshape(size, size)


def tbplw_shuffle_to_cont(origin: np.ndarray) -> np.ndarray:
    """shuffle matrix in TBPLW model to the order implemented in CONT

    Note: matrix can be hamk and bands[i]. The order now in CONT is:
    layer index, g index, A/B index. The order now in TBPLW is:
    layer index, A/B index, g index.

    ATTENTION: you need to make sure the order of Glist is consistent in both methods.

    Args:
        origin (np.ndarray): original matrix in the order of layer index, A/B index, g index.

    Returns:
        np.ndarray: output matrix in the order of layer index, g index, A/B index.
    """

    size = origin.shape[0]
    num_g = size//4
    origin_expanded = origin.reshape(2, 2, num_g, 2, 2, num_g)
    origin_shuffled = np.moveaxis(origin_expanded, [1, 4], [2, 5])

    return origin_shuffled.reshape(size, size)
