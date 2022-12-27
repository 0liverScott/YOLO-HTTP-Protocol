from typing import Tuple


def ycc_to_rgb(y: int, cr: int, cb: int) -> Tuple[int, int, int]:
    """Convert YUV values to RGB colors according to the BT.601 standard."""
    r = (298.082 * y / 256) + (408.583 * cr / 256) - 222.921
    g = (298.082 * y / 256) - (100.291 * cb / 256) - (208.120 * cr / 256) + 135.576
    b = (298.082 * y / 256) + (516.412 * cb / 256) - 276.836

    r = max(min(round(r), 255), 0)
    g = max(min(round(g), 255), 0)
    b = max(min(round(b), 255), 0)

    return (r, g, b)
