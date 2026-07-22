"""The taste space: fixed users, mobile creators, and the top-K recommender rule.

Nothing here learns. This module just defines the world and the one rule that
connects users to creators.
"""

from dataclasses import dataclass, field

import numpy as np

# Each entry: (center_x, center_y, sigma, n_users, label)
DEFAULT_CLUSTERS = [
    (0.0, 0.0, 1.0, 700, "mainstream"),
    (3.0, 3.0, 0.3, 50, "niche_a"),
    (-3.0, 2.0, 0.3, 30, "niche_b"),
]


@dataclass
class WorldConfig:
    clusters: list = field(default_factory=lambda: list(DEFAULT_CLUSTERS))
    sigma_utility: float = 1.0   # RBF width: "how far a user will walk"
    bounds: tuple = (-5.0, 5.0)  # creators are clipped to this box


def sample_users(cfg: WorldConfig, rng: np.random.Generator):
    """Draw the user population once. They never move again.

    Returns
    -------
    positions : (N, 2) float array
    labels    : (N,) array of str  -- which cluster each user came from.
                KEEP THIS. It is what lets you measure niche welfare
                separately from mainstream welfare later.

    TODO: for each (cx, cy, sigma, n, label) in cfg.clusters, draw n points
    from a Gaussian centered at (cx, cy) with that sigma, then concatenate.
    """
    raise NotImplementedError


def init_creators(n_creators: int, cfg: WorldConfig, rng: np.random.Generator,
                  mode: str = "spread"):
    """Place the creators at round 0.

    mode="spread"     -> uniform over cfg.bounds. Diversity starts high, so you
                         can watch it collapse. This is the default experiment.
    mode="mainstream" -> all clumped near (0, 0). Flips the question to:
                         can exploration ever *find* the niches? Run this second.

    Returns (M, 2) float array.
    """
    raise NotImplementedError


def pairwise_sq_dists(users: np.ndarray, creators: np.ndarray) -> np.ndarray:
    """Squared euclidean distance from every user to every creator.

    Returns (N, M). Broadcast users[:, None, :] against creators[None, :, :],
    square, sum over the last axis. No loops.
    """
    raise NotImplementedError


def utility(sq_dists: np.ndarray, sigma: float) -> np.ndarray:
    """RBF utility in [0, 1]:  exp(-d^2 / (2 * sigma^2)).

    Note this is strictly decreasing in distance, so it produces the *same*
    top-K as ranking by raw distance. It exists to make the welfare numbers
    bounded and comparable, not to change who gets recommended.

    Returns (N, M).
    """
    raise NotImplementedError


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Which creators each user actually sees.

    scores : (N, M) higher is better (use the utility matrix)
    Returns (N, M) boolean mask with exactly k True per row.

    Use np.argpartition along axis=1 -- you don't need a full sort.
    """
    raise NotImplementedError


def exposure(mask: np.ndarray) -> np.ndarray:
    """Payoff for each creator this round = number of users who saw it.

    Returns (M,) int array. This is the only feedback a creator ever gets.
    """
    raise NotImplementedError
