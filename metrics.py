"""What you measure each round. These four numbers are the whole result.

The one that matters most is welfare_by_group: average welfare can look
perfectly healthy while the niche tail quietly starves.
"""

import numpy as np


def content_diversity(creators: np.ndarray) -> float:
    """Mean pairwise distance between creators.

    Falls as the creator cloud contracts. This is your headline curve.
    Use the upper triangle only -- don't let the zero diagonal drag it down.
    """
    raise NotImplementedError


def user_welfare(util: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-user welfare given what they were actually shown.

    util : (N, M) RBF utilities
    mask : (N, M) boolean top-K mask

    Returns (N,). Start with the best single item shown -- max of util where
    mask is True. (Sum-over-shown is the alternative; if you switch, say so in
    the README, because it changes how K affects the numbers.)
    """
    raise NotImplementedError


def welfare_by_group(welfare: np.ndarray, labels: np.ndarray) -> dict:
    """Mean welfare per user cluster, e.g.
    {"mainstream": 0.81, "niche_a": 0.12, "niche_b": 0.09}

    This is the plot that makes the point.
    """
    raise NotImplementedError


def exposure_gini(exposure: np.ndarray) -> float:
    """Gini coefficient of exposure across creators. 0 = perfectly equal,
    ~1 = one creator takes everything.

    Tells you whether the ecosystem is becoming winner-take-all economically,
    not just aesthetically. Sort ascending, then use the standard formula:
        (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n     with i starting at 1
    Guard against sum(x) == 0.
    """
    raise NotImplementedError


def round_record(round_idx: int, creators: np.ndarray, util: np.ndarray,
                 mask: np.ndarray, exposure_counts: np.ndarray,
                 labels: np.ndarray) -> dict:
    """Bundle one round's metrics into a flat dict, ready to append to a list
    and hand to pd.DataFrame at the end.

    Suggested keys: round, diversity, welfare_mean, welfare_<each label>, gini.
    """
    raise NotImplementedError
