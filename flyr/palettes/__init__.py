""" Module containing the palette definitions flyr supports and the function
    with which to use it.

    Specifically see the `map_colors` function, which uses any of the palettes
    defined in the `palettes` dictionary.
"""

from typing import List, Dict, Union, Tuple

import numpy as np
from nptyping import Array

from .cividis import cividis
from .copper import copper
from .gist_earth import gist_earth
from .gist_rainbow import gist_rainbow
from .hot import hot
from .inferno import inferno
from .jet import jet
from .magma import magma
from .ocean import ocean
from .plasma import plasma
from .rainbow import rainbow
from .terrain import terrain
from .turbo import turbo
from .viridis import viridis


palettes: Dict[str, List[Tuple[int, int, int]]] = {
    "cividis": cividis,
    "copper": copper,
    "gist_earth": gist_earth,
    "gist_rainbow": gist_rainbow,
    "hot": hot,
    "inferno": inferno,
    "jet": jet,
    "magma": magma,
    "ocean": ocean,
    "plasma": plasma,
    "rainbow": rainbow,
    "terrain": terrain,
    "turbo": turbo,
    "viridis": viridis,
}


def map_colors(
    thermal: Array[float, ..., ...],
    palette: Union[str, List[Tuple[int, int, int]]] = "jet",
) -> Array[np.uint8, ..., ..., 3]:
    """ Colors a normalized image array.

        Parameters
        ----------
        thermal: Array[float, ..., ...]
            A normalized array, that is all values are between 0 and 1.

        Returns
        -------
        Array[np.uint8, ..., ..., 3]
            A 3D array with channels as its last dimension, containing integers
            between 0 and 255.
    """
    if isinstance(palette, list):
        colors = palette
    elif isinstance(palette, str):
        colors = palettes[palette]

    colored = np.zeros(thermal.shape + (3,), dtype=np.uint8)
    num_colors = len(colors)
    for idx, color in enumerate(colors):
        lower = idx / num_colors
        upper = (idx + 1) / num_colors
        colored[(lower <= thermal) & (thermal <= upper)] = color

    return colored
