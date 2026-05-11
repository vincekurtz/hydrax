import jax.numpy as jnp
import pytest

from hydrax.utils.spline import (
    compute_b_spline_matrix,
    interp_akima,
    interp_bspline,
)


@pytest.mark.parametrize(
    "x",
    [
        jnp.linspace(1.0, 5.0, 5),  # unit spacing
        jnp.array([0.0, 0.5, 1.0, 1.5]),  # arbitrary spacing
    ],
    ids=["unit_spacing", "half_spacing"],
)
def test_akima_passes_through_knots(x: jnp.ndarray) -> None:
    """Akima spline reproduces the waypoints at every knot."""
    m_pts = x.shape[0]
    y = jnp.array(
        [[0.0, 0.0], [1.0, -1.0], [0.5, 2.0], [2.0, 0.5], [1.5, -0.5]]
    )[:m_pts]

    result = interp_akima(x, x, y[None, ...])  # (1, M, D)
    assert jnp.allclose(result[0], y, atol=1e-5), (
        f"Expected {y}, got {result[0]}"
    )


def test_akima_beats_linear_on_sin_wave() -> None:
    """Akima fit of a smooth signal beats linear interpolation."""
    m_pts = 9
    tk = jnp.linspace(0.0, 2.0 * jnp.pi, m_pts)
    y = jnp.sin(tk)[:, None]  # (M, 1)

    tq = jnp.linspace(0.0, 2.0 * jnp.pi, 200)
    akima_vals = interp_akima(tq, tk, y[None, ...])[0, :, 0]
    truth = jnp.sin(tq)
    linear = jnp.interp(tq, tk, y[:, 0])

    akima_mse = jnp.mean((akima_vals - truth) ** 2)
    linear_mse = jnp.mean((linear - truth) ** 2)
    assert akima_mse < linear_mse, (
        f"Akima MSE {akima_mse:.6f} should beat linear MSE {linear_mse:.6f}"
    )


def test_b_spline_matrix_partition_of_unity() -> None:
    """Each row of the B-spline basis matrix sums to one."""
    for degree, m_pts in [(2, 3), (2, 4), (3, 4), (3, 5)]:
        knots = jnp.arange(m_pts + degree + 1, dtype=jnp.float32)
        mat = compute_b_spline_matrix(knots, degree, num_points=20)
        assert mat.shape == (20, m_pts)
        row_sums = jnp.sum(mat, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5), (
            f"Partition of unity failed for degree={degree}, M={m_pts}: "
            f"row sums {row_sums}"
        )


def test_interp_bspline_shape() -> None:
    """interp_bspline returns the correct output shape."""
    m_pts, degree, num_points = 4, 2, 20
    knot_vec = jnp.arange(m_pts + degree + 1, dtype=jnp.float32)
    bmat = compute_b_spline_matrix(knot_vec, degree, num_points)

    B, D = 3, 2
    waypoints = jnp.ones((B, m_pts, D))
    result = interp_bspline(bmat, waypoints)
    assert result.shape == (B, num_points, D)
