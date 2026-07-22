# Creator-Arena


A simulator of content creators competing for exposure under a top-K
recommender. Creators adapt using bandit feedback - they see only their own
exposure, never the user distribution - and the question is what kind of
content ecosystem that grows over time.

## Motivation

This simulator is inspired by the line of work on strategic content creators
under recommendation, in particular:

- Yao, F., Li, C., Nekipelov, D., Wang, H., & Xu, H. (2023).
  *How Bad is Top-K Recommendation under Competing Content Creators?*
  ICML 2023 (Oral). https://arxiv.org/abs/2302.01971
- Yao, F., Li, C., Nekipelov, D., Wang, H., & Xu, H. (2024).
  *Human vs. Generative AI in Content Creation Competition: Symbiosis or
  Conflict?* ICML 2024. https://arxiv.org/abs/2402.15467
- Yao, F., et al. *User Welfare Optimization in Recommender Systems with
  Competing Content Creators.* https://arxiv.org/abs/2404.18319

The spatial-competition framing goes back further, to Hotelling (1929) and
its later applications to recommender systems.

**Scope.** This is a simplified agent-based sandbox built to develop intuition
for the setting, not a reproduction of any published experiment and not a
theoretical contribution. The papers above analyze welfare loss via Price of
Anarchy under a Random Utility choice model with no-regret creators; this
repository instead asks what the dynamics look like empirically when those
assumptions are relaxed - deterministic user choice, small K, and a fixed
epsilon-greedy learner. Any divergence from their results is expected to come
from those relaxations rather than from a contradiction of them.

## Setup

Users are fixed points in a 2D taste space, drawn from a Gaussian mixture:
one dense mainstream cluster plus two small niche clusters. Creators are
mobile points in the same space. Each round the recommender shows every user
their K nearest creators; a creator's payoff is how many users saw it.

## Install

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python run.py --demo     # single run -> figures/drift.gif
python run.py --sweep    # 90 runs -> results.csv + figures/
```

## Results

TODO -- fill in after the sweep. Table of final diversity and niche welfare
by K and creator count.

| K | creators | final diversity | mainstream welfare | niche welfare | gini |
|---|----------|-----------------|--------------------|---------------|------|
|   |          |                 |                    |               |      |

## Findings

To do

## Limitations

- Users are static
- Utility is pure proximity, there is no novelty term, no diminishing returns on
  seeing similar content.
- Creators relocate instantly and at zero cost; real repositioning is slow
  and expensive.
- 2D is not what real embedding spaces look like; high-dimensional taste
  spaces have far more room to differentiate.
- The recommender is fixed and non-learning, so this measures creator
  adaptation alone, not the two-sided dynamic.
