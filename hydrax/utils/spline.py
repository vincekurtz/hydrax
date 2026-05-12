from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from interpax import Akima1DInterpolator, interp1d
from jax import vmap

InterpMethodType = Literal["zero", "linear", "cubic", "akima"]
InterpFuncType = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


"""We define the interpolation functions here so they're picklable for async."""


@partial(vmap, in_axes=(None, None, 0))
def interp_zero(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Zero-order spline interpolation."""
    # for a zero-order spline, take the "next" knot as the control
    # ex: tq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #     tk = [0.0, 0.25, 0.5]
    #     inds = [0, 0, 0, 1, 1, 2]  # searchsorted trick does this
    #     interp_func(tq, tk, knots) = knots[:, inds]
    return knots[jnp.searchsorted(tk, tq, side="right") - 1]


@partial(vmap, in_axes=(None, None, 0))
def interp_linear(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Linear spline interpolation."""
    return interp1d(tq, tk, knots, method="linear", extrap=True)


@partial(vmap, in_axes=(None, None, 0))
def interp_cubic(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Cubic spline interpolation."""
    return interp1d(tq, tk, knots, method="cubic2", extrap=True)


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
        method: The interpolation method to use. Can be "zero", "linear",
            "cubic", or "akima".

    Returns:
        interp_func: The interpolation function.
    """
    if method == "zero":
        interp_func = interp_zero
    elif method == "linear":
        interp_func = interp_linear
    elif method == "cubic":
        interp_func = interp_cubic
    elif method == "akima":
        interp_func = interp_akima
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            "Expected one of ['zero', 'linear', 'cubic', 'akima']."
        )
    return interp_func


def interp_akima(
    tq: jax.Array, tk: jax.Array, knots: jax.Array
) -> jax.Array:
    """Akima spline interpolation over a batch of waypoint sequences.

    Uses ``interpax.Akima1DInterpolator`` (Akima, "A New Method of
    Interpolation and Smooth Curve Fitting Based on Local Procedures",
    J. ACM 17(4), 1970, pp. 589--602).

    Args:
        tq: Query times, shape ``(H,)``.
        tk: Knot positions, shape ``(M,)``.
        knots: Waypoint values, shape ``(B, M, D)``.

    Returns:
        Interpolated values of shape ``(B, H, D)``.
    """

    def _one(c: jax.Array) -> jax.Array:
        return Akima1DInterpolator(tk, c, check=False)(tq)

    return vmap(_one)(knots)


def compute_b_spline_matrix(
    x: jax.Array, degree: int, num_points: int
) -> jax.Array:
    """Build the B-spline basis matrix on the valid parameter domain.

    Constructs the Cox-de Boor basis matrix evaluated at ``num_points``
    uniformly-spaced parameter values inside the valid B-spline domain
    ``[x[degree], x[-degree-1]]``, so every row satisfies the partition
    of unity.

    Args:
        x: The knot vector, shape ``(M + degree + 1,)``.
        degree: The B-spline degree (``>= 2``).
        num_points: Number of evaluation points across the valid domain.

    Returns:
        Basis matrix of shape ``(num_points, M)`` whose rows sum to 1.
    """
    t_start = x[degree]
    t_end = x[-degree - 1]
    # Tiny inward shrink avoids the right-open boundary at t == t_end.
    eps = (t_end - t_start) * 1e-7
    t_values = jnp.linspace(t_start, t_end - eps, num_points)

    b = jnp.where(
        (x[:-1] <= t_values[:, None]) & (t_values[:, None] < x[1:]),
        1.0,
        0.0,
    )

    for d in range(1, degree + 1):
        left_d1, left_d2 = x[d:-1], x[: -d - 1]
        b_left = jnp.where(
            left_d1 > left_d2,
            (
                (t_values[:, None] - left_d2)
                / jnp.where(left_d1 > left_d2, left_d1 - left_d2, 1.0)
            )
            * b[:, :-1],
            0.0,
        )
        right_d1, right_d2 = x[d + 1 :], x[1:-d]
        b_right = jnp.where(
            right_d1 > right_d2,
            (
                (right_d1 - t_values[:, None])
                / jnp.where(right_d1 > right_d2, right_d1 - right_d2, 1.0)
            )
            * b[:, 1:],
            0.0,
        )
        b = b_left + b_right

    return b


def interp_bspline(
    bmat: jax.Array, knots: jax.Array
) -> jax.Array:
    """B-spline interpolation via a pre-computed basis matrix.

    Args:
        bmat: Basis matrix of shape ``(H, M)`` from
            :func:`compute_b_spline_matrix`.
        knots: Waypoint values, shape ``(B, M, D)``.

    Returns:
        Interpolated values of shape ``(B, H, D)``.
    """
    return jnp.einsum("bmd,hm->bhd", knots, bmat)
