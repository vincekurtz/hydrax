from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from interpax import interp1d
from jax import jit, vmap

InterpMethodType = Literal["zero", "linear", "cubic"]
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
        method: The interpolation method to use. Can be "zero", "linear", or
            "cubic".

    Returns:
        interp_func: The interpolation function.
    """
    if method == "zero":
        interp_func = interp_zero
    elif method == "linear":
        interp_func = interp_linear
    elif method == "cubic":
        interp_func = interp_cubic
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            "Expected one of ['zero', 'linear', 'cubic']."
        )
    return interp_func


# ---------------------------------------------------------------------------
# Akima and B-spline primitives used by the MTP controller.
#
# These live alongside the standard interp_* helpers so MTP can share the
# spline module with every other algorithm. They are deliberately pure
# functions (no closure over a controller instance) so they JIT-compile
# once per shape and stay picklable for async use.
# ---------------------------------------------------------------------------


@jit
def poly_akima(x: jax.Array, c: jax.Array) -> jax.Array:
    """Compute modified-Akima cubic spline coefficients through waypoints.

    Builds the per-segment cubic polynomial coefficients of a modified Akima
    spline (Akima 1970, with the symmetric mean fallback for collinear
    sections) through ``M`` waypoints in ``D`` dimensions. The returned
    coefficients are evaluated by :func:`poly_interpolation`.

    Args:
        x: Knot positions, shape ``(M,)``. Must be uniformly spaced.
        c: Waypoint values, shape ``(M, D)``.

    Returns:
        Polynomial coefficients of shape ``(M-1, 4, D)``. Segment ``i``
        stores ``[d_i, c_i, b_i, a_i]`` so that the cubic on
        ``[x[i], x[i+1])`` is ``d_i*t^3 + c_i*t^2 + b_i*t + a_i``.
    """
    dx = jnp.diff(x)
    m_pts, dim = c.shape[0], c.shape[1]
    n_seg = m_pts - 1

    # Slopes between breakpoints, with a safe inverse spacing.
    safe_dx = jnp.where(dx == 0, 1.0, dx)
    dxr = jnp.where(dx == 0, 0.0, 1.0 / safe_dx)[:, None]
    y_l, y_r = c[:-1], c[1:]
    slopes = (y_r - y_l) * dxr

    # M==2: the Akima padded-slope cascade reads uninitialised entries with
    # only one real slope; fall back to a single linear segment with
    # endpoint derivatives both equal to that slope.
    if n_seg == 1:
        dydx_l = slopes * dxr
        dydx_r = slopes * dxr
        ai = y_l
        bi = dydx_l
        ci = 3.0 * slopes - 2.0 * dydx_l - dydx_r
        di = -2.0 * slopes + dydx_l + dydx_r
        a = jnp.stack([di, ci, bi, ai], axis=-1)
        scale = dxr[:, None] ** jnp.arange(4)[::-1]
        a = a / scale
        return jnp.moveaxis(a, -1, 1)

    # Pad slope array with two boundary extrapolations on each side.
    m = jnp.zeros((n_seg + 4, dim))
    m = m.at[2 : 2 + n_seg].set(slopes)
    m = m.at[1].set(2.0 * m[2] - m[3])
    m = m.at[0].set(2.0 * m[1] - m[2])
    m = m.at[-2].set(2.0 * m[-3] - m[-4])
    m = m.at[-1].set(2.0 * m[-2] - m[-3])

    # Akima derivative weights (modified form: symmetric mean fallback when
    # both weights vanish, avoiding the original 0/0 division).
    dm = jnp.abs(jnp.diff(m, axis=0))
    pm = jnp.abs(m[1:] + m[:-1])
    f1 = dm[2:] + 0.5 * pm[2:]
    f2 = dm[:-2] + 0.5 * pm[:-2]
    m2 = m[1:-2]
    m3 = m[2:-1]
    f12 = f1 + f2
    nonzero = f12 > 1e-9 * jnp.max(jnp.abs(f12), initial=0.0)
    dydx = jnp.where(
        nonzero,
        (f1 * m2 + f2 * m3) / jnp.where(nonzero, f12, 1.0),
        0.5 * (m2 + m3),
    )

    dydx_l = dydx[:-1] * dxr
    dydx_r = dydx[1:] * dxr
    ai = y_l
    bi = dydx_l
    ci = 3.0 * slopes - 2.0 * dydx_l - dydx_r
    di = -2.0 * slopes + dydx_l + dydx_r
    a = jnp.stack([di, ci, bi, ai], axis=-1)

    # Rescale coefficients for the actual segment spacing (dx != 1).
    scale = dxr[:, None] ** jnp.arange(4)[::-1]
    a = a / scale
    return jnp.moveaxis(a, -1, 1)


@partial(jit, static_argnames=("num_points",))
def poly_interpolation(a: jax.Array, num_points: int = 5) -> jax.Array:
    """Evaluate a batch of Akima spline segments at uniform query points.

    Uses a single ``einsum`` with a precomputed Vandermonde matrix to
    evaluate every segment of every batch element in parallel.

    Args:
        a: Polynomial coefficients of shape ``(B, M-1, 4, D)`` produced by
           :func:`poly_akima` (batched along the first axis).
        num_points: Number of evaluation points per segment.

    Returns:
        Interpolated values of shape ``(B, (M-1) * num_points, D)``.
    """
    b, n_seg, _, dim = a.shape
    t = jnp.linspace(0.0, 1.0, num_points + 1)[:-1]
    powers = jnp.arange(3, -1, -1)
    t_powers = t[:, None] ** powers[None, :]
    spline = jnp.einsum("bspd,tp->bstd", a, t_powers)
    return spline.reshape(b, n_seg * num_points, dim)


def compute_b_spline_matrix(
    x: jax.Array, degree: int, num_points: int
) -> jax.Array:
    """Build the B-spline basis matrix on the valid parameter domain.

    Constructs the Cox-de Boor basis matrix evaluated at ``num_points``
    uniformly-spaced parameter values inside the *valid* B-spline domain
    ``[x[degree], x[-degree-1]]``, so every row satisfies the partition
    of unity. Multiplying ``(num_points, M)`` against ``(B, M, D)`` knot
    samples then yields a clean batched einsum.

    Args:
        x: The knot vector, shape ``(M + degree + 1,)``.
        degree: The B-spline degree (``>= 2``).
        num_points: Number of evaluation points across the valid domain.

    Returns:
        Basis matrix of shape ``(num_points, M)`` whose rows sum to 1.
    """
    t_start = x[degree]
    t_end = x[-degree - 1]
    # Tiny inward shrink avoids the right-open ``< x[i+1]`` boundary at
    # ``t == t_end`` that would otherwise fire two intervals at once.
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
