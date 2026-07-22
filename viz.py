"""Plotting. The snapshot GIF is the artifact people will actually look at.

Snapshot cadence is controlled by SimConfig.snapshot_every in run.py -- every
5 rounds gives you ~60 frames over a 300-round run, which is a smooth GIF
without a huge file.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; write straight to files
import matplotlib.pyplot as plt
import numpy as np

CLUSTER_COLORS = {
    "mainstream": "#B5D4F4",
    "niche_a": "#9FE1CB",
    "niche_b": "#F5C4B3",
}
CREATOR_COLOR = "#BA7517"


def plot_square(ax, users: np.ndarray, labels: np.ndarray,
                creators: np.ndarray, round_idx: int, bounds: tuple) -> None:
    """Draw one frame of the world onto an existing axis.

    Users as small faded dots colored by cluster, creators as larger opaque
    markers on top. Keep axis limits FIXED to bounds on every frame -- if
    matplotlib autoscales, the GIF will appear to zoom and the drift you're
    trying to show becomes invisible.

    Put the round number in the title so frames are self-labeling.
    """
    raise NotImplementedError


def save_snapshot(users, labels, creators, round_idx, bounds,
                  outdir: Path) -> Path:
    """Render one frame with plot_square and save it as frame_{round:04d}.png.

    Zero-pad the round number -- otherwise sorted() gives you 1, 10, 100, 2
    and the GIF plays out of order. Returns the path written.
    """
    raise NotImplementedError


def make_gif(frame_paths: list, out_path: Path, fps: int = 10) -> Path:
    """Stitch saved frames into an animated GIF.

    Easiest route: pillow.
        from PIL import Image
        frames = [Image.open(p) for p in frame_paths]
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)

    (matplotlib.animation.FuncAnimation with PillowWriter also works if you'd
    rather not write frames to disk at all.)
    """
    raise NotImplementedError


def plot_metric_curves(df, metric: str, group_by: str, out_path: Path) -> Path:
    """Metric over rounds, one line per value of group_by (usually 'k'),
    with mean +/- std ribbon across seeds.

    The +/- std band is what makes it a result rather than an anecdote --
    a single unseeded run proves nothing.
    """
    raise NotImplementedError


def plot_welfare_gap(df, out_path: Path) -> Path:
    """Mainstream welfare vs each niche welfare over rounds, same axes.

    If the collapse happens, this is the figure you paste into the email.
    """
    raise NotImplementedError
