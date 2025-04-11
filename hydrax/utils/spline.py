from typing import Callable, Literal

import jax
import jax.numpy as jnp
from interpax import interp1d
from jax import vmap

InterpMethodType = Literal["zero", "linear", "cubic"]
InterpFuncType = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def get_interp_func(method: InterpMethodType) -> InterpFuncType:
    """Get the 1D interpolation function based on the specified method.

    In particular, the function will have signature
        u_traj = interp_func(tq, tk, knots),
    where
        * tq is a 1D array of query times of shape (H,)
        * tk is a 1D array of knot times of shape (num_knots,),
        * knots is an array of shape (num_rollouts, num_knots), and
        * u_traj is the batch of interpolated trajectories of shape
            (num_rollouts, H).
    Here, we expect H to be the number of control time steps over some horizon T
    in seconds.

    Args:
        method: The interpolation method to use. Can be "zero", "linear", or
            "cubic".

    Returns:
        interp_func: The interpolation function.
    """
    if method == "zero":
        # for a zero-order spline, take the "next" knot as the control
        # ex: tq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        #     tk = [0.0, 0.25, 0.5]
        #     inds = [0, 0, 0, 1, 1, 2]  # searchsorted trick does this
        #     interp_func(tq, tk, knots) = knots[:, inds]
        interp_func = vmap(
            lambda tq, tk, knots: knots[
                jnp.searchsorted(tk, tq, side="right") - 1
            ],
            in_axes=(None, None, 0),
        )
    elif method in ["linear", "cubic"]:
        # we use "cubic" to mean natural splines, not local, which corresponds
        # to the "cubic2" method in interpax
        # https://github.com/f0uriest/interpax/blob/163c348925167f82a3658a094458cb2608f189ff/interpax/_spline.py#L47
        method = "cubic2" if method == "cubic" else "linear"
        interp_func = vmap(
            lambda tq, tk, knots: interp1d(tq, tk, knots, method=method),
            in_axes=(None, None, 0),
        )
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            "Expected one of ['zero', 'linear', 'cubic']."
        )
    return interp_func
