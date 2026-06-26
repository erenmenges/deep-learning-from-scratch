import numpy as np

def model_linear(phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (phi[1] * x) + phi[0]

def loss_linear(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((model_linear(phi,x) - y) ** 2)

def grad_linear(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(np.array([(model_linear(phi,x) - y) * 2, (model_linear(phi,x) - y) * 2 * x]), axis=1)

def gradient_descent_linear(phi0: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float, n_iters: int) -> tuple:
    loss_history = []
    loss_history.append(loss_linear(phi0, x, y))
    for _ in range(n_iters):
        phi0 = phi0 - alpha * grad_linear(phi0, x, y)
        loss_history.append(loss_linear(phi0, x, y))
    return [phi0, loss_history]

def momentum_step(phi: np.ndarray, g: np.ndarray, m: np.ndarray, alpha: float, beta: float) -> tuple:
    m_next = (beta * m) + ((1 - beta) * g)
    phi = phi - (alpha * m_next)
    return (phi, m_next)

def adam_step(phi: np.ndarray, g: np.ndarray, m: np.ndarray, v:np.ndarray, t: int, alpha: float, beta: float, gamma: float, eps: float) -> tuple:
    m_next = (beta * m) + ((1 - beta) * g)
    v_next = (gamma * v) + ((1 - gamma) * (g**2))

    m_hat_next = m_next / (1 - (beta**t))
    v_hat_next = v_next / (1 - (gamma**t))
    phi = phi - alpha * (m_hat_next/(np.sqrt(v_hat_next) + eps))
    return (phi, m_next, v_next)

def model_gabor(phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sin(phi[0] + phi[1] * 0.06 * x) * np.exp(-1 * (((phi[0] + phi[1] * 0.06 * x)** 2)/32.0))

def loss_gabor(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((model_gabor(phi,x) - y) ** 2)

def grad_gabor(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = phi[0] + 0.06 * phi[1] * x

    d1 = (np.cos(a) * np.exp(-1 * ((a ** 2)/32.0))) + (np.sin(a) * np.exp(-1 * ((a ** 2)/32.0)) * (-2 * (a/32.0)))
    d2 = (np.cos(a) * (0.06 * x) * np.exp(-1 * ((a ** 2)/32.0))) + (np.sin(a) * np.exp(-1 * ((a ** 2)/32.0)) * (-2 * (a/32.0)) * (0.06 * x))

    d1 = 2 * (model_gabor(phi,x) - y) * d1
    d2 = 2 * (model_gabor(phi,x) - y) * d2
    grad = np.sum([d1, d2], axis=1)
    return grad


def sgd_gabor(phi0: np.ndarray, x: np.ndarray, y: np.ndarray, alpha0: float, batch_size: int, n_epochs: int, decay_factor: float, decay_every: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    epoch_losses = np.zeros((n_epochs,))

    alpha = alpha0
    phi = phi0
    for e in range(n_epochs):
        perm = rng.permutation(x.shape[0])

        for start in range(0, N, batch_size):
            x_batch = x[perm[start : start + batch_size]]
            y_batch = y[perm[start : start + batch_size]]
            phi = phi - alpha * grad_gabor(phi, x_batch, y_batch)
        
        if ((e + 1) % decay_every == 0):
            alpha *= decay_factor
        
        epoch_losses[e] = loss_gabor(phi, x, y)
    
    return (phi, epoch_losses)
