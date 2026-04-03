import numpy as np
import matplotlib.pyplot as plt

def original_cost(x, func='exp'):
    """The original "spiky" cost function."""
    if func == 'exp':
        return -np.exp(-np.square(x-0.7)/1e-4)
    elif func == 'square':
        return np.where((x < 0.75) & (x > 0.65), 0.0, 1.0)
    elif func == 'quadratic':
        return (x - 0.7)**2
    elif func == 'vshape':
        return np.abs(x - 0.7)
    elif func == 'double_well':
        return -0.8 * np.exp(-np.square(x - 0.62) / 5e-4) \
               -1.0 * np.exp(-np.square(x - 0.78) / 2e-3)
    else:
        raise ValueError("Unsupported cost function type")


def sample_uniform(num_samples=200, shift_range=0.04):
    """Sample uniformly from the given range."""
    return np.random.uniform(-shift_range, shift_range, num_samples)

def sample_gaussian(num_samples=200, std=0.005):
    """Sample from a Gaussian distribution."""
    return np.random.normal(0, std, num_samples)

def sample_shifts(sample_func="uniform", num_samples=200, shift_range=0.04, std=0.005):
    """Sample shifts based on the specified sampling function."""
    if sample_func == "uniform":
        return sample_uniform(num_samples=num_samples, shift_range=shift_range)
    elif sample_func == "gaussian":
        return sample_gaussian(num_samples=num_samples, std=std)
    else:
        raise ValueError("Unsupported sampling function type")


def average_randomization(x, dx, func='exp'):
    """Shift the cost by back and forth over a uniform distribution
    and take the average.

    This simulates the "average case" domain randomization technique.
    """
    costs = np.array([original_cost(x + shift, func=func) for shift in dx])
    return np.mean(costs)

def worst_case_randomization(x, dx, func='exp'):
    """Shift the cost and take the worst case (max cost).

    This simulates the conservative "worst case" domain randomization technique.
    """
    costs = np.array([original_cost(x + shift, func=func) for shift in dx])
    return np.max(costs)

def best_case_randomization(x, dx, func='exp'):
    """Shift the cost and take the best case (min cost).

    This simulates the optimistic "best case" domain randomization technique.
    """
    costs = np.array([original_cost(x + shift, func=func) for shift in dx])
    return np.min(costs)

cost_func = "double_well"
num_samples = 200
shift_ranges = [0.01, 0.04, 0.08]
x = np.linspace(0.5, 0.9, 1000)
y = original_cost(x, func=cost_func)

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharey=True)

for ax, sr in zip(axes, shift_ranges):
    dx = sample_shifts(sample_func="uniform", num_samples=num_samples, shift_range=sr)

    y_avg = np.array([average_randomization(xi, dx=dx, func=cost_func) for xi in x])
    y_worst = np.array([worst_case_randomization(xi, dx=dx, func=cost_func) for xi in x])
    y_best = np.array([best_case_randomization(xi, dx=dx, func=cost_func) for xi in x])

    ax.plot(x, y, label='Original Cost', color='black', linestyle='--')
    ax.plot(x, y_avg, label='Average')
    ax.plot(x, y_worst, label='Worst Case')
    ax.plot(x, y_best, label='Best Case')
    ax.set_title(f"shift_range = {sr}")
    ax.set_ylabel('J(x)')
    if ax is axes[0]:
        ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('x')
plt.tight_layout()
plt.show()