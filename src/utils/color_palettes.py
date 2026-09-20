from __future__ import annotations


SCI_SOFT_COLORS = [
    "#DEECF6",
    "#AFC8E2",
    "#E2F2CD",
    "#B6DAA7",
    "#F9D5D5",
    "#EF98A1",
    "#FBE3C0",
    "#FBC99A",
    "#E8E0EF",
    "#C2B1D7",
    "#CFE8E0",
    "#8FC7BD",
    "#F6E6A8",
    "#D9C27F",
    "#D6E4F0",
    "#89A8C9",
]


def get_sci_soft_colors(n: int | None = None) -> list[str]:
    """Return the soft scientific color palette."""
    if n is None:
        return SCI_SOFT_COLORS.copy()
    if n <= 0:
        return []
    repeats = (n + len(SCI_SOFT_COLORS) - 1) // len(SCI_SOFT_COLORS)
    return (SCI_SOFT_COLORS * repeats)[:n]
