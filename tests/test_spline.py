import jax.numpy as jnp
import pytest

from hydrax.utils.spline import get_interp_func


def test_zero_interp_func() -> None:
    """Tests the correctness of the zero-order interpolation function."""
    func = get_interp_func("zero")

    tq = jnp.linspace(0.0, 1.0, 11)
    tk = jnp.array([0.0, 0.5, 1.0])
    knots = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    out = func(tq, tk, knots)
    expected = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0],
        ]
    )
    assert jnp.allclose(out, expected), f"Expected {expected}, got {out}!"


def test_linear_interp_func() -> None:
    """Tests the correctness of the linear interpolation function."""
    func = get_interp_func("linear")

    tq = jnp.linspace(0.0, 1.0, 11)
    tk = jnp.array([0.0, 0.5, 1.0])
    knots = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    out = func(tq, tk, knots)
    expected = jnp.array(
        [
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0],
        ]
    )
    assert jnp.allclose(out, expected), f"Expected {expected}, got {out}!"


@pytest.mark.skip(reason="Failure due to interpax bug.")
def test_cubic_interp_func() -> None:
    """Tests the correctness of the cubic interpolation function.

    NOTE: this test is currently failing, but I think there's some deep bug in
    the interp1d implementation. See this open issue I opened:
    https://github.com/f0uriest/interpax/issues/87
    """
    func = get_interp_func("cubic")

    tq = jnp.linspace(0.0, 2.0 * jnp.pi, 10001)
    tk = jnp.linspace(0.0, 2.0 * jnp.pi, 101)
    f = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=0)
    knots = f(tk)
    out = func(tq, tk, knots)
    assert jnp.allclose(out, f(tq), rtol=1e-6, atol=1e-5), (
        f"Expected {f(tq)}, got {out}!"
    )


if __name__ == "__main__":
    test_zero_interp_func()
    test_linear_interp_func()
    test_cubic_interp_func()
    print("All tests passed!")
