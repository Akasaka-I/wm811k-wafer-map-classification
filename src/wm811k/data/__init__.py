"""Data inspection, preprocessing, splitting, and loading."""

CLASS_NAMES = (
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
)

CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
