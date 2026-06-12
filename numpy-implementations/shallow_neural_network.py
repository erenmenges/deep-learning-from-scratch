import numpy as np

# synthetic data
rng = np.random.default_rng(3)
theta  = rng.standard_normal((3,2))     # rows [theta_d0, theta_d1]
omega  = rng.standard_normal(3)
omega0 = float(rng.standard_normal())
x      = np.linspace(-3, 3, 10)


# code
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)  ### for each element if z is negative then maximum will be 0, and if positive, then z


def shallow_forward(X: np.ndarray, Theta: np.ndarray, theta0: np.ndarray, Omega: np.ndarray, omega0: np.ndarray) -> np.ndarray:
    """Multivariate forward"""
    pre_activations = (X @ Theta.T) + theta0[None ,:]  ### shape: (N, D_i) @ (D_i, D) + (1, D) = (N, D)
    activations = relu(pre_activations)
    y_hat = omega0[None, :] + activations @ Omega.T  ### shape: (N, D) @ (D, D_o) + (1, D_o) = (N, D_o)
    return y_hat

def net_forward(x: np.ndarray, theta: np.ndarray, omega: np.ndarray, omega0: np.ndarray) -> np.ndarray:
    "scalar forward"
    pre_activations = x[:, None] @ (theta[:,-1])[None, :] + (theta[:, 0])[None, :]  ### shape: (N,1) @ (1, D) + (1, D) = (N, D)
    activations = relu(pre_activations)
    y_hat = activations @ omega.T + omega0
    return y_hat

def joints(theta: np.ndarray) -> np.ndarray:
    return (-1 * theta[:, 0])/theta[:, -1]

def activation_pattern(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    bool_mask = np.zeros((len(x), len(theta)), dtype=bool)
    for i, elem in enumerate(x):
        pre_activations = np.array([elem]) @ (theta[:,-1])[None, :] + (theta[:, 0])[None, :] ### shape (1,1) @ (1,D) + (1,D) = (1,D)
        bool_mask[i] = pre_activations >= 0
    return bool_mask

def region_slopes(theta: np.ndarray, omega: np.ndarray, x_grid: np.ndarray) -> list:
    bool_mask = activation_pattern(x_grid, theta)
    region_start_indexes = [x_grid[0]]
    for i,row in enumerate(bool_mask):
        if (i != 0) and ((row != bool_mask[i-1]).any()):
            region_start_indexes.append(x_grid[i])
    region_start_indexes = np.array(region_start_indexes)
    bool_mask = activation_pattern(region_start_indexes, theta)
    bool_mask = np.where(bool_mask, 1, 0)
    
    
    slopes_matrix = (omega * theta[:, -1])[None, :] * bool_mask ### shape: (1, D) * (number_of_regions, D) = (number_of_regions, D)
    slopes = np.sum(slopes_matrix, axis=-1)
    return slopes