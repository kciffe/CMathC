from __future__ import annotations


SCI_HEIGHT_COLORS = [
    "#D7EEF4",
    "#C9E8DB",
    "#D7EAC2",
    "#EFE6B5",
    "#F4D2A8",
    "#F0B8A8",
    "#E6B1BD",
    "#D3B5D8",
    "#BFC0E3",
    "#A9CBE4",
    "#90C8D2",
    "#78BEB8",
    "#6DB3A7",
    "#6DA6C8",
    "#6E93BD",
    "#747EB0",
]


def get_sci_height_colors(n: int | None = None) -> list[str]:
    """Return a muted ordered palette for height-layer boxplots."""
    return _take_colors(SCI_HEIGHT_COLORS, n)


def _take_colors(colors: list[str], n: int | None = None) -> list[str]:
    if n is None:
        return colors.copy()
    if n <= 0:
        return []
    repeats = (n + len(colors) - 1) // len(colors)
    return (colors * repeats)[:n]
