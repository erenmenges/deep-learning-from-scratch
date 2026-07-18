import numpy as np

def relu(f: np.ndarray) -> np.ndarray:
    return np.maximum(f, 0)

def forward(x: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, dict]:
    num_k = len(params["b"]) - 1
    f = [None] * (num_k + 1)
    h = [None] * (num_k + 1)
    f[0] = params["b"][0] + params["W"][0] @ x
    for k in range(1, num_k + 1):
        h[k] = relu(f[k-1])
        f[k] = params["b"][k] + params["W"][k] @ h[k]
    cache = {"f": f, "h": h}
    return (f[num_k], cache)

def backward(x: np.ndarray, y: np.ndarray, params: np.ndarray, cache: dict) -> dict:
    num_k = len(params["b"]) - 1

    f_grads = [None] * (num_k + 1)
    f_grads[num_k] = 2 * (cache["f"][num_k] - y)

    grads = {"W": [None] * (num_k + 1), "b": [None] * (num_k + 1)}

    for k in range(num_k, 0, -1):
        grads["W"][k] = f_grads[k][:,None] @ cache["h"][k][None, :]
        grads["b"][k] = f_grads[k]
        f_grads[k-1] = ((cache["f"][k-1]) > 0).astype(float) * (params["W"][k].T @ f_grads[k])
    
    grads["W"][0] = f_grads[0][:, None] @ x[None, :]
    grads["b"][0] = f_grads[0]

    return grads

def batch_grads(X: np.ndarray, Y: np.ndarray, params: np.ndarray) -> np.ndarray:
    cache_list = []
    for x_i in X:
        cache_list.append(forward(x_i, params)[1])

    grads_list = []
    for i, x_i in enumerate(X):
        grads_list.append(backward(x_i, Y[i], params, cache_list[i]))

    num_k = len(params["b"]) - 1
    output = {"W": [None] * (num_k + 1), "b": [None] * (num_k + 1)}


    for k in range(num_k + 1):
        sum_W = np.zeros(grads_list[0]["W"][k].shape)
        sum_b = np.zeros(grads_list[0]["b"][k].shape)
        for grad in grads_list:
            sum_W += grad["W"][k] 
            sum_b += grad["b"][k] 
        output["W"][k] = sum_W
        output["b"][k] = sum_b
    return output
    