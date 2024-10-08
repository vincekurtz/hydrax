from hydra.tasks.pendulum import Pendulum


def test_pendulum() -> None:
    """Make sure we can instantiate the Pendulum task."""
    task = Pendulum()
    assert isinstance(task, Pendulum)


if __name__ == "__main__":
    test_pendulum()
