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


SCI_DEEP_COLORS = [
    "#1F77B4",  # blue
    "#D95F02",  # orange
    "#E6AB02",  # golden
    "#6A3D9A",  # purple
    "#1B9E77",  # teal green
    "#17A9D6",  # cyan blue
    "#E7298A",  # magenta
    "#66A61E",  # green
    "#A6761D",  # brown gold
    "#7570B3",  # indigo
    "#E64B35",  # red
    "#4DBBD5",  # sky blue
]


def get_sci_height_colors(n: int | None = None) -> list[str]:
    """Return a muted ordered palette for height-layer boxplots."""
    return _take_colors(SCI_HEIGHT_COLORS, n)


def get_sci_deep_colors(n: int | None = None) -> list[str]:
    """Return a deeper scientific palette for line/profile plots."""
    return _take_colors(SCI_DEEP_COLORS, n)


def _take_colors(colors: list[str], n: int | None = None) -> list[str]:
    if n is None:
        return colors.copy()
    if n <= 0:
        return []
    repeats = (n + len(colors) - 1) // len(colors)
    return (colors * repeats)[:n]
