import copy
import numpy as np

# code from my batch_backprop.py
def relu(f: np.ndarray) -> np.ndarray:
    return np.maximum(f, 0)

def forward(X: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, dict]:
    num_k = len(params["b"]) - 1
    F = [None] * (num_k + 1)
    H = [None] * (num_k + 1)
    F[0] = params["b"][0] + X @ params["W"][0].T
    
    for k in range(1, num_k + 1):
        H[k] = relu(F[k-1])
        F[k] = params["b"][k] + H[k] @ params["W"][k].T 
    cache = {"F": F, "H": H}
    return (F[num_k], cache)

def backward(X: np.ndarray, Y: np.ndarray, params: np.ndarray, cache: dict) -> dict:
    num_k = len(params["b"]) - 1
    F = cache["F"]
    H = cache["H"]
    W = params["W"]
    b = params["b"]

    dF = [None] * (num_k + 1)
    dF[num_k] = 2 * (F[num_k] - Y)

    grads = {"W": [None] * (num_k + 1), "b": [None] * (num_k + 1)}

    for k in range(num_k, 0, -1):
        grads["W"][k] = dF[k].T @ H[k]
        grads["b"][k] = dF[k].sum(axis=0)
        dF[k-1] = (F[k-1] > 0).astype(float) * (dF[k] @ W[k])
    
    grads["W"][0] = dF[0].T @ X
    grads["b"][0] = dF[0].sum(axis=0)

    return grads


# training code
def make_batches(N: int, batch_size: int, rng) -> list[int]:
    indices = np.array(range(N))
    rng.shuffle(indices)  ## shuffled indices

    batches = []
    for i in range(0, N, batch_size):
        batches.append(indices[i: i + batch_size])
    return batches

def train_sgd(X: np.ndarray, Y: np.ndarray, params: dict, alpha: float, batch_size: int, num_epochs: int, seed: int) -> tuple[dict, list[float]]:
    rng = np.random.default_rng(seed)
    training_params = copy.deepcopy(params)
    W = training_params["W"]
    b = training_params["b"]
    

    losses = []
    for _ in range(num_epochs):
        batches = make_batches(X.shape[0], batch_size, rng)
        for batch in batches:
            X_b = X[batch]
            Y_b = Y[batch]
            cache = forward(X_b, training_params)[1]
            grads = backward(X_b, Y_b, training_params, cache)
            gW = grads["W"]
            gb = grads["b"]
            for k in range(len(gW)):
                W[k] = W[k] - alpha * gW[k]
                b[k] = b[k] - alpha * gb[k]
        losses.append(((forward(X, training_params)[0] - Y)**2).sum(axis=1).sum(axis=0))
    final_params = {"W": W, "b": b}
    return (final_params, losses)


def train_momentum(X: np.ndarray, Y: np.ndarray, params: dict, alpha: float,
                   beta: float, batch_size: int, num_epochs: int, seed: int) -> tuple[dict, list[float]]:
    rng = np.random.default_rng(seed)
    training_params = copy.deepcopy(params)
    W = training_params["W"]
    b = training_params["b"]

    losses = []

    momentum_W = [0] * len(W)
    momentum_b = [0] * len(b)

    for _ in range(num_epochs):
        batches = make_batches(X.shape[0], batch_size, rng)
        for batch in batches:
            X_b = X[batch]
            Y_b = Y[batch]
            cache = forward(X_b, training_params)[1]
            grads = backward(X_b, Y_b, training_params, cache)
            gW = grads["W"]
            gb = grads["b"]
            for k in range(len(gW)):
                momentum_W[k] = (beta * momentum_W[k]) + ((1 - beta) * gW[k])
                W[k] = W[k] - alpha * momentum_W[k]
                momentum_b[k] = (beta * momentum_b[k]) + ((1 - beta) * gb[k])
                b[k] = b[k] - alpha * momentum_b[k]
            
        losses.append(((forward(X, training_params)[0] - Y)**2).sum(axis=1).sum(axis=0))
    final_params = {"W": W, "b": b}
    return (final_params, losses)