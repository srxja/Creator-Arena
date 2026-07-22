"""Entry point: one run = n_rounds of the loop. One sweep = many runs.

    python run.py --demo        # single run, writes snapshots + gif
    python run.py --sweep       # full grid, writes results.csv + figures
"""

import argparse
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import world
import creators as creators_mod
import metrics
import viz

FIGDIR = Path("figures")
FRAMEDIR = Path("figures/frames")


@dataclass
class SimConfig:
    k: int = 3
    n_creators: int = 20
    n_rounds: int = 300
    seed: int = 0
    epsilon: float = 0.1
    step_size: float = 0.05
    creator_init: str = "spread"
    snapshot_every: int = 5   # frame cadence for the GIF; 0 disables snapshots


def simulate(cfg: SimConfig, wcfg: world.WorldConfig, snapshot: bool = False):
    """Run one simulation.

    Returns
    -------
    history : list of dict   -- one metrics.round_record per round
    frames  : list of Path   -- snapshot images (empty if snapshot is False)

    The loop body, in order:
        1. sq = world.pairwise_sq_dists(users, pop.positions)
        2. util = world.utility(sq, wcfg.sigma_utility)
        3. mask = world.top_k_mask(util, cfg.k)
        4. expo = world.exposure(mask)
        5. history.append(metrics.round_record(...))
        6. if snapshot and t % cfg.snapshot_every == 0: viz.save_snapshot(...)
        7. pop.step(expo)          # learn from this round, then move

    Always snapshot round 0 and the final round regardless of cadence -- the
    before/after pair is the thing worth looking at.
    """
    rng = np.random.default_rng(cfg.seed)
    users, labels = world.sample_users(wcfg, rng)
    positions = world.init_creators(cfg.n_creators, wcfg, rng, cfg.creator_init)
    pop = creators_mod.CreatorPopulation(
        positions, rng, epsilon=cfg.epsilon, step_size=cfg.step_size,
        bounds=wcfg.bounds,
    )

    history, frames = [], []
    raise NotImplementedError  # TODO: the loop above


def sweep(ks=(1, 3, 10), n_creators=(5, 20, 50), seeds=range(10)):
    """Run the full grid and return a tidy DataFrame.

    Every row = one round of one run, with k / n_creators / seed as columns,
    so you can groupby(['k', 'round']).agg(['mean', 'std']) for the ribbons.

    90 runs at ~1s each. If it's slow, shrink the user count before you shrink
    the seed count -- seeds are what make the result credible.
    """
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="single run with snapshots + gif")
    ap.add_argument("--sweep", action="store_true",
                    help="full grid -> results.csv + figures")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--creators", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    FRAMEDIR.mkdir(parents=True, exist_ok=True)
    wcfg = world.WorldConfig()

    if args.demo:
        cfg = SimConfig(k=args.k, n_creators=args.creators,
                        n_rounds=args.rounds, seed=args.seed)
        history, frames = simulate(cfg, wcfg, snapshot=True)
        pd.DataFrame(history).to_csv("demo_history.csv", index=False)
        viz.make_gif(frames, FIGDIR / "drift.gif")

    if args.sweep:
        df = sweep()
        df.to_csv("results.csv", index=False)
        viz.plot_metric_curves(df, "diversity", "k", FIGDIR / "diversity.png")
        viz.plot_metric_curves(df, "gini", "k", FIGDIR / "gini.png")
        viz.plot_welfare_gap(df, FIGDIR / "welfare_gap.png")


if __name__ == "__main__":
    main()
