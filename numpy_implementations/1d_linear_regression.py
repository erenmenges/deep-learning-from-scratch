import numpy as np

# synthetic data
true_w = np.array([0.5, 1.5])
SIGMA = 0.3

def make_data(N, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, size=N)
    y = true_w[0] + true_w[1] * x + rng.normal(scale=SIGMA, size=N)
    return x, y

x_test, y_test = make_data(500, seed=99999)          # fixed test set

w0_grid = np.linspace(-1.0, 3.0, 81)             # step 0.05
w1_grid = np.linspace(-1.0, 3.0, 81)
sizes  = [3, 8, 20, 100, 200, 500]
seeds  = range(30)                                    # average over 30 training draws per size

# code

def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[0]+ (w[1] * x)

def loss(x: np.ndarray, y: np.ndarray, w :np.ndarray) -> float:
    return np.sum((predict(x,w) - y)**2)

def loss_grad(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.array([np.sum(2 * (predict(x,w) - y)),
             np.sum(2 * (predict(x,w) - y) * x)])

def fit_grid(x: np.ndarray, y: np.ndarray, w0_grid: np.ndarray, w1_grid: np.ndarray) -> np.ndarray:
    w0 = w0_grid[:, None, None] ### shape (G0, 1, 1)
    w1 = w1_grid[None, :, None] ### shape (1, G1, 1)
    xx = x[None, None, :] ### shape (1, 1, I)
    yy = y[None, None, :] ### shape (1, 1, I)

    loss_grid = (((w0 + w1 * xx) - yy)**2).sum(axis=-1)

    min_row, min_col = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    return np.array([w0_grid[min_row], w1_grid[min_col]])

def experiment_with_different_sizes(x_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_param_dist = np.zeros((len(sizes),))
    mean_test_loss = np.zeros((len(sizes),))
    for i, size in enumerate(sizes):
        training_runs_param_dists = np.zeros(len(seeds))
        training_runs_test_losses = np.zeros(len(seeds))
        for j, seed in enumerate(seeds):
            x, y = make_data(size, seed)
            best_w = fit_grid(x, y, w0_grid, w1_grid)
            training_runs_param_dists[j] = np.linalg.norm(best_w - true_w)
            training_runs_test_losses[j] = loss(x_test, y_test, best_w)
        mean_param_dist[i] = np.mean(training_runs_param_dists)
        mean_test_loss[i] = np.mean(training_runs_test_losses)
    return (mean_param_dist, mean_test_loss)

mean_param_dist, mean_test_loss = experiment_with_different_sizes(x_test, y_test)
        
# plot
import matplotlib.pyplot as plt
plt.plot(sizes, mean_test_loss, marker="o")
plt.xlabel("training set size N")
plt.ylabel("mean test loss across 30 runs")
plt.title("Test loss vs training set size")
plt.show()
