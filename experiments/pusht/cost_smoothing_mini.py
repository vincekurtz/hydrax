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

def original_cost(x):
    """Single well cost function."""
    return -np.exp(-np.square(x - 0.7) / 1e-4)


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


def average_randomization(x, dx):
    """Shift the cost by back and forth over a uniform distribution
    and take the average.

    This simulates the "average case" domain randomization technique.
    """
    costs = np.array([original_cost(x + shift) for shift in dx])
    return np.mean(costs)

def worst_case_randomization(x, dx):
    """Shift the cost and take the worst case (max cost).

    This simulates the conservative "worst case" domain randomization technique.
    """
    costs = np.array([original_cost(x + shift) for shift in dx])
    return np.max(costs)

def best_case_randomization(x, dx):
    """Shift the cost and take the best case (min cost).

    This simulates the optimistic "best case" domain randomization technique.
    """
    costs = np.array([original_cost(x + shift) for shift in dx])
    return np.min(costs)

num_samples = 2000
sr = 0.01
x = np.linspace(0.5, 0.9, 1000)
y = original_cost(x)
dx = sample_shifts(sample_func="uniform", num_samples=num_samples, shift_range=sr)

y_worst = np.array([worst_case_randomization(xi, dx=dx) for xi in x])
y_best = np.array([best_case_randomization(xi, dx=dx) for xi in x])

color_pes = np.array([216, 27, 96]) / 255
color_opt = np.array([255, 193, 7]) / 255

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharey=True)

# Top: nominal + pessimistic
ax1.plot(x, y, label=r'$J(u)$', color='black', linestyle='--')
ax1.plot(x, y_worst, label=r'$\bar{J}_{\mathrm{pes}}$', color=color_pes)
ax1.set_yticks([])
ax1.set_yticklabels([])
ax1.set_ylabel(r'$J(\mathbf{u})$')
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_xlim(0.65, 0.75)
ax1.set_xlabel(r'$\mathbf{u}$')

# Bottom: nominal + optimistic
ax2.plot(x, y, label=r'$J(u)$', color='black', linestyle='--')
ax2.plot(x, y_best, label=r'$\bar{J}_{\mathrm{opt}}$', color=color_opt)
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.set_ylabel(r'$J(\mathbf{u})$')
ax2.set_xlim(0.665, 0.735)
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_xlabel(r'$\mathbf{u}$')
plt.tight_layout()
plt.show()