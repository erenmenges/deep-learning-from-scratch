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
    


from sklearn.datasets import make_moons

rng = np.random.default_rng(123)

# ---- Build a fixed network: D_in=2, hidden [5,4,3], D_out=1  (K=3 hidden layers)
dims = [2, 5, 4, 3, 1]          # D_0 ... D_{K+1}, with K=3
K = len(dims) - 2               # number of hidden layers = 3

def _init_params():
    W, b = {}, {}
    for k in range(K + 1):       # k = 0..K
        out_d, in_d = dims[k + 1], dims[k]
        W[k] = rng.standard_normal((out_d, in_d)) * 0.5
        b[k] = rng.standard_normal(out_d) * 0.5
    return {'W': W, 'b': b}

params = _init_params()

# ---- Small dataset
Xnp, ynp = make_moons(n_samples=16, noise=0.1, random_state=0)
X = Xnp.astype(float)
Y = ynp.astype(float).reshape(-1, 1)

def _batch_loss(params):
    total = 0.0
    for i in range(X.shape[0]):
        fK, _ = forward(X[i], params)
        total += float(np.sum((fK - Y[i]) ** 2))
    return total

# ---- (a) Gradient check the SUMMED batch gradient against finite differences
analytic = batch_grads(X, Y, params)

def _num_grad_param(arr_ref, eps=1e-5):
    g = np.zeros_like(arr_ref)
    it = np.nditer(arr_ref, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        old = arr_ref[idx]
        arr_ref[idx] = old + eps; Lp = _batch_loss(params)
        arr_ref[idx] = old - eps; Lm = _batch_loss(params)
        arr_ref[idx] = old
        g[idx] = (Lp - Lm) / (2 * eps)
    return g

for k in range(K + 1):
    numW = _num_grad_param(params['W'][k])
    numb = _num_grad_param(params['b'][k])
    assert np.allclose(analytic['W'][k], numW, atol=1e-4), \
        f"W[{k}] grad mismatch: max err {np.max(np.abs(analytic['W'][k]-numW)):.2e}"
    assert np.allclose(analytic['b'][k], numb, atol=1e-4), \
        f"b[{k}] grad mismatch: max err {np.max(np.abs(analytic['b'][k]-numb)):.2e}"

# ---- (b) The summed gradient must point DOWNHILL.
#      A single fixed-alpha step is NOT guaranteed to lower a non-convex loss
#      (it depends on curvature), so instead we test the two things that ARE
#      guaranteed for a correct gradient:
#        (i)  the directional derivative of the loss along -grad equals -||grad||^2
#        (ii) a sufficiently small step along -grad decreases the loss.
g = batch_grads(X, Y, params)
gnorm2 = sum(float(np.sum(g['W'][k]**2)) + float(np.sum(g['b'][k]**2))
             for k in range(K + 1))
assert gnorm2 > 0, "summed gradient is exactly zero -- something is wrong"

t = 1e-5
def _shift(sign):
    return {'W': {k: params['W'][k] + sign * t * (-g['W'][k]) for k in range(K + 1)},
            'b': {k: params['b'][k] + sign * t * (-g['b'][k]) for k in range(K + 1)}}

# central-difference directional derivative of L along d = -grad
dir_deriv = (_batch_loss(_shift(+1)) - _batch_loss(_shift(-1))) / (2 * t)
assert dir_deriv < 0, f"summed gradient is not a descent direction: {dir_deriv:.4e}"
assert np.isclose(dir_deriv, -gnorm2, rtol=1e-3, atol=1e-3), \
    f"directional derivative {dir_deriv:.4e} should equal -||grad||^2 {-gnorm2:.4e}"

# a small actual step must decrease the loss
t_small = 1e-4
L0 = _batch_loss(params)
p_step = {'W': {k: params['W'][k] - t_small * g['W'][k] for k in range(K + 1)},
          'b': {k: params['b'][k] - t_small * g['b'][k] for k in range(K + 1)}}
assert _batch_loss(p_step) < L0, \
    f"small GD step did not decrease the loss: {L0:.4f} -> {_batch_loss(p_step):.4f}"

print("Problem 3 passed.")