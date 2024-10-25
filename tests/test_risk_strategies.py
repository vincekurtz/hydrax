import jax
import jax.numpy as jnp

from hydrax.risk import (
    AverageCost,
    BestCase,
    ExponentialWeightedAverage,
    WorstCase,
)


def test_risk() -> None:
    """Quick sanity check on various risk strategies."""
    rng = jax.random.key(0)

    n, m = 10, 20
    costs = jax.random.normal(rng, (n, m))

    avg = AverageCost().combine_costs(costs)
    assert avg.shape == (m,)

    worst = WorstCase().combine_costs(costs)
    assert worst.shape == (m,)

    best = BestCase().combine_costs(costs)
    assert best.shape == (m,)

    weighted = ExponentialWeightedAverage(2.0).combine_costs(costs)
    assert weighted.shape == (m,)

    assert jnp.all(avg <= worst)
    assert jnp.all(avg >= best)
    assert jnp.all(weighted <= worst)
    assert jnp.all(weighted >= best)
    assert jnp.all(weighted >= avg)


if __name__ == "__main__":
    test_risk()
