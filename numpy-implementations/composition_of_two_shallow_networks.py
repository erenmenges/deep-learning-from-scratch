import numpy as np

# code

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0,z)

def composed_sequential(x: np.ndarray, omega0: np.ndarray, omega1: np.ndarray,
                         phi: np.ndarray, omega0p: np.ndarray, omega1p: np.ndarray, phip: np.ndarray) -> np.ndarray:
    # network 1
    pre_activations = omega0[None,:] + (x[:, None] @ omega1[None, :]) ## shape: (1, D) + ((B,1) @ (1, D)) = (B, D)
    activations = relu(pre_activations)
    y_hat = phi[0] + (activations @ phi[1:][:, None])  ## shape: (1,1) + ((B, D) @ (D, 1)) = (B,1)

    # network 2
    pre_activations_p = omega0p[None, :] + (y_hat @ omega1p[None, :]) ## shape: (1, D) + ((B,1) @ (1, D)) = (B, D)
    activations_p = relu(pre_activations_p)
    y_prime = phip[0] + (activations_p @ phip[1:][:, None])  ## shape: (1,1) + ((B, D) @ (D, 1)) = (B,1)
    return y_prime

def composed_as_deep(x: np.ndarray, omega0: np.ndarray, omega1: np.ndarray,
                         phi: np.ndarray, omega0p: np.ndarray, omega1p: np.ndarray, phip: np.ndarray) -> np.ndarray:
    
    pre_activations = omega0[None, :] + (x[:, None] @ omega1[None, :]) ## shape: (1, D) + ((B,1) @ (1, D)) = (B, D)
    activations = relu(pre_activations)

    psi1 = omega1p[None, :] * phi[1:][:, None]  ## shape: (1, D) * (D, 1) = (D, D)
    psi0 = (omega1p[None, :] * phi[0]) + omega0p[None, :] ## shape: (1,D) * (1,1) + (1,D) = (1, D)

    pre_activations_p = psi0 + (activations @ psi1)  ## shape: ((1,D) + (B,D) @ (D, D)) = (B, D)
    activations_p = relu(pre_activations_p)
    y_hat = phip[0] + (activations_p @ phip[1:][:, None])
    return y_hat

def init_params(dims: list, seed: int) -> tuple[list, list]:
    Omegas = []
    Betas = []
    rng = np.random.default_rng(seed)
    for i in range(len(dims)-1):
        omega = rng.normal(size=(dims[i+1], dims[i]))
        beta = rng.normal(size=(dims[i+1],))
        Omegas.append(omega)
        Betas.append(beta)
    return (Omegas, Betas)

def forward(X: np.ndarray, Omegas: list, Betas: list) -> np.ndarray:
    current = None
    past = X
    for i, (omega, beta) in enumerate(zip(Omegas, Betas)):
        if i != len(Betas) - 1:
            current = relu(beta[:, None] + (omega @ past))
        else: 
            current = beta[:, None] + (omega @ past)
        past = current
    return current


def count_params(K: int, D: int) -> int:
    return (3 * D) + 1 + ((K - 1) * D * (D + 1))

def max_regions(K: int, D: int) -> int:
    return (D + 1) ** K

def make_random_net(K: int, D: int, seed: int) -> tuple:
    dims = [D] * K
    dims.append(1)
    dims.insert(0,1)
    return init_params(dims, seed)

