"""How creators adapt.

Each creator sees ONLY its own exposure history -- never the user distribution,
never another creator's payoff. It picks among a few candidate moves and learns
which ones pay. That makes this a bandit, not an optimization.

The environment is non-stationary: standing still can stop paying simply
because a competitor moved. That is the interesting part.
"""

import numpy as np

# The action set. Index i means "step in direction MOVES[i]".
MOVES = np.array([
    [0.0, 0.0],   # stay
    [1.0, 0.0],   # east
    [-1.0, 0.0],  # west
    [0.0, 1.0],   # north
    [0.0, -1.0],  # south
])


class CreatorPopulation:
    """All M creators, vectorized. No per-creator Python loop in the hot path."""

    def __init__(self, positions: np.ndarray, rng: np.random.Generator,
                 epsilon: float = 0.1, step_size: float = 0.05,
                 bounds: tuple = (-5.0, 5.0)):
        self.positions = positions.astype(float)
        self.rng = rng
        self.epsilon = epsilon
        self.step_size = step_size
        self.bounds = bounds

        m, a = len(positions), len(MOVES)
        self.q = np.zeros((m, a))       # running mean payoff per (creator, move)
        self.counts = np.zeros((m, a))  # times each move has been tried
        self.last_action = None         # (M,) int, set by select_actions

    def select_actions(self) -> np.ndarray:
        """Epsilon-greedy over MOVES, independently per creator.

        Returns (M,) int array of chosen move indices. Also store it on
        self.last_action so update() knows what to credit.

        TODO: with prob epsilon pick a uniform random move, else argmax(self.q).
        Vectorize with a single rng.random(m) < epsilon mask.

        Optional upgrade once this works: swap in UCB
        (q + c * sqrt(log(t) / counts)) and compare the two in your README.
        """
        raise NotImplementedError

    def apply_actions(self, actions: np.ndarray) -> None:
        """Move each creator step_size along its chosen direction, then clip
        to self.bounds so nobody wanders off the map."""
        raise NotImplementedError

    def update(self, rewards: np.ndarray) -> None:
        """Incremental mean update for the action each creator just took.

        rewards : (M,) exposure counts from this round.

        For each creator i with action a = self.last_action[i]:
            counts[i, a] += 1
            q[i, a] += (reward[i] - q[i, a]) / counts[i, a]

        TODO: do this with fancy indexing, not a for-loop over creators.
        Consider normalizing reward by n_users so epsilon/step tuning
        transfers across population sizes.
        """
        raise NotImplementedError

    def step(self, rewards_from_last_round: np.ndarray) -> None:
        """One full creator turn: learn from last round, then move.

        Order matters -- update() must see the reward that the PREVIOUS action
        earned, before select_actions() overwrites self.last_action.
        """
        raise NotImplementedError
