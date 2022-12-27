from nptyping import Array

import numpy as np


def by_minmax(
    min_t: float, max_t: float, thermal: Array[float, ..., ...]
) -> Array[float, ..., ...]:
    """ Normalize the given array to be between 0 and 1 using the given minimum
        and maximum values.

        Parameters
        ----------
        min_t: float
            The value to clip the input array to and which should map to 0.
        max_t: float
            The value to clip the input array to and which should map to 1.
        thermal: Array[float, ..., ...]
            The array to normalize.

        Returns
        -------
        Array[float, ..., ...]
            An array with all values between 0 and 1
    """
    err_msg = f"Minimum value {min_t} should be smaller than maximum value {max_t}"
    assert min_t < max_t, err_msg
    thermal = np.clip(thermal, min_t, max_t)
    thermal = (thermal - min_t) / (max_t - min_t)
    return thermal


def by_percentiles(
    min_p: float, max_p: float, thermal: Array[np.float64, ..., ...]
) -> Array[np.float64, ..., ...]:
    """ Normalizes the given thermal array using the values at the given minimum
        and maximum percentiles.

        Parameters
        ----------
        min_p: float
            Value between 0 and 1 that is lower than `max_p`. Indicates the
            percentile from which to take the minimum normalization value.
        max_p: float
            Value between 0 and 1 that is greater than `min_p`. Indicates the
            percentile from which to take the maximum normalization value.
        thermal: Array[np.float64, ..., ...]
            Array to normalize.

        Returns
        -------
        Array[np.float64, ..., ...]
            The normalized array, all np.float64 values between 0 and 1.
    """
    assert 0.0 <= min_p < max_p <= 1.0
    return by_minmax(
        np.percentile(thermal, int(min_p * 100)),
        np.percentile(thermal, int(max_p * 100)),
        thermal,
    )
