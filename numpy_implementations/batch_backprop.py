import numpy as np

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