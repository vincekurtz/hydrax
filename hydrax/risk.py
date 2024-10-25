from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class RiskStrategy(ABC):
    """An abstract risk strategy interface.

    A risk strategy defines how we combine costs from different domains with
    domain randomization. For example, a very risk-averse strategy might take
    the worst-case cost over all randomizations, while a risk-seeking strategy
    might take the best-case cost.
    """

    @abstractmethod
    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Combine costs from different randomizations.

        Args:
            costs: rollout costs, first axis is the randomization index.

        Returns:
            The combined cost (scalar).
        """
        pass


class AverageCost(RiskStrategy):
    """Average cost risk strategy.

    This is the standard expectation over randomizations that is often used in
    reinforcement learning.
    """

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the average cost over all randomizations."""
        return jnp.mean(costs, axis=0)
