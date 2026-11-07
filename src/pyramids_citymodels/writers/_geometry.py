"""Small geometry helpers shared by the mesh-oriented writers (OBJ, glTF, 3D Tiles).

These stay ``private`` to the ``writers`` package: they are pure coordinate utilities with no IR
knowledge, lifted here so the OBJ, glTF and 3D-Tiles writers share one definition rather than each
carrying its own copy.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any


def open_ring(ring: Any) -> list[Any]:
    """Drop a ring's closing duplicate vertex (``last == first``), so a face is not degenerate.

    Args:
        ring: A sequence of coordinates whose first and last may repeat (a closed GML ring).

    Returns:
        The ring as a list with the trailing duplicate removed when present, else a copy as-is.

    Examples:
        - A closed ring loses its repeated last vertex:
            ```python
            >>> open_ring([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

            ```
        - An already-open ring is returned unchanged (as a new list):
            ```python
            >>> open_ring([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

            ```
    """
    if len(ring) > 1 and tuple(ring[0]) == tuple(ring[-1]):
        return list(ring[:-1])
    return list(ring)


def xyz(coordinate: Any) -> tuple[float, float, float]:
    """A coordinate as an ``(x, y, z)`` float triple, filling ``z = 0.0`` for a 2-D point.

    Args:
        coordinate: A sequence of two or three numeric components.

    Returns:
        The coordinate as an ``(x, y, z)`` float tuple, with ``z = 0.0`` when 2-D.

    Examples:
        - A 3-D coordinate is returned as floats:
            ```python
            >>> xyz((1, 2, 3))
            (1.0, 2.0, 3.0)

            ```
        - A 2-D coordinate gains a zero ``z``:
            ```python
            >>> xyz((4.5, 6.5))
            (4.5, 6.5, 0.0)

            ```
    """
    x = float(coordinate[0])
    y = float(coordinate[1])
    z = float(coordinate[2]) if len(coordinate) > 2 else 0.0
    return (x, y, z)
