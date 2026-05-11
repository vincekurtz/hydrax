import jax.numpy as jnp

from hydrax.utils.spline import (
    compute_b_spline_matrix,
    poly_akima,
    poly_interpolation,
)


def test_akima_passes_through_knots() -> None:
    """Akima spline reproduces the waypoints at every knot."""
    m_pts = 5
    x = jnp.linspace(1.0, m_pts, m_pts)
    rng_seed = 0
    y = jnp.array(
        [[0.0, 0.0], [1.0, -1.0], [0.5, 2.0], [2.0, 0.5], [1.5, -0.5]]
    )
    a = poly_akima(x, y)  # (m_pts - 1, 4, 2)

    # Each segment evaluated at t=0 should reproduce the left waypoint.
    left = a[:, 3, :]
    assert jnp.allclose(left, y[:-1], atol=1e-6), (
        f"Expected segment starts {y[:-1]}, got {left}; seed={rng_seed}"
    )


def test_akima_beats_linear_on_sin_wave() -> None:
    """Akima fit of a smooth signal beats linear interpolation."""
    m_pts = 9
    x = jnp.linspace(1.0, m_pts, m_pts)
    sample_t = jnp.linspace(0.0, 2.0 * jnp.pi, m_pts)
    y = jnp.sin(sample_t)[:, None]

    a = poly_akima(x, y)
    fine = poly_interpolation(a[None, ...], num_points=20)[0, :, 0]

    # Reference signal at the same fine times.
    seg_t = jnp.linspace(0.0, 1.0, 20 + 1)[:-1]
    fine_t = jnp.concatenate(
        [
            sample_t[i] + seg_t * (sample_t[i + 1] - sample_t[i])
            for i in range(m_pts - 1)
        ]
    )
    truth = jnp.sin(fine_t)
    linear = jnp.interp(fine_t, sample_t, y[:, 0])

    akima_mse = jnp.mean((fine - truth) ** 2)
    linear_mse = jnp.mean((linear - truth) ** 2)
    assert akima_mse < linear_mse, (
        f"Akima MSE {akima_mse:.4f} should beat linear MSE {linear_mse:.4f}"
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


if __name__ == "__main__":
    test_akima_passes_through_knots()
    test_akima_beats_linear_on_sin_wave()
    test_b_spline_matrix_partition_of_unity()
    print("All MTP spline tests passed!")
