"""
style.py - one shared look for every Matplotlib chart in the project.

Colours: a fixed 8-colour categorical palette (colour-blind checked), always
assigned in the same order, plus a single-hue blue ramp for magnitudes.
Text stays in neutral ink; only marks carry colour.
"""
import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in fixed order (cluster 0 -> slot 1, ...). Max 8 categories.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"      # "other" / de-emphasised marks
# Sequential blue ramp, light (near zero) -> dark (high).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "civic_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply():
    """Set Matplotlib defaults: light surface, thin recessive axes, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def color_for(i: int) -> str:
    """Categorical colour for category index i (fixed order, never cycled)."""
    if i >= len(CATEGORICAL):
        raise ValueError("More than 8 categories - fold extras into 'Other'.")
    return CATEGORICAL[i]
