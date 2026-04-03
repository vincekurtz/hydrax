import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

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
        return -0.6 * np.exp(-np.square(x - 0.62) / 5e-4) \
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

    ax.plot(x, y, label=r'$J(u)$', color='black', linestyle='--')
    ax.plot(x, y_worst, label=r'$\bar{J}_{\mathrm{pes}}$', color=np.array([216, 27, 96]) / 255)
    ax.plot(x, y_avg, label=r'$\bar{J}_{\mathrm{avg}}$', color=np.array([30, 136, 229]) / 255)
    ax.plot(x, y_best, label=r'$\bar{J}_{\mathrm{opt}}$', color=np.array([255, 193, 7]) / 255)
    title_labels = {0.01: r"Small $\delta$", 0.04: r"Moderate $\delta$", 0.08: r"Large $\delta$"}
    ax.set_title(title_labels[sr])
    ax.set_ylabel(r'$J(u)$')
    if ax is not axes[-1]:
        ax.set_xticklabels([])
    ax.set_xlim(0.5, 0.9)
    if ax is axes[0]:
        ax.legend(ncol=4, draggable=True, columnspacing=1.0, handletextpad=0.4)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(r'$u$', fontsize=18)
plt.tight_layout()
plt.show()