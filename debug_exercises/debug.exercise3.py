import numpy as np


def relu(f):
    return np.maximum(f, 0)

def forward(X, params):
    """Standard forward pass, our usual convention (batch-first, H @ W.T).
    Returns (output, cache); cache['F'] is the list of pre-activations F[0..last]."""
    num_k = len(params["b"]) - 1
    F = [None] * (num_k + 1)
    H = [None] * (num_k + 1)
    F[0] = params["b"][0] + X @ params["W"][0].T
    for k in range(1, num_k + 1):
        H[k] = relu(F[k-1])
        F[k] = params["b"][k] + H[k] @ params["W"][k].T
    return F[num_k], {"F": F, "H": H}

# --- network config: a deep network, mirroring the Figure 7.7 experiment ---
D_in       = 200     # input width (first layer maps 200 -> 100)
Dh         = 100     # hidden units per layer
NUM_LAYERS = 50      # number of stacked weight matrices

def build_params(init_fn, rng):
    """Build the network's params dict. `init_fn(fan_in, fan_out, rng)` decides
    how each weight matrix is initialized. Biases are all zero (as in Fig 7.7)."""
    W, b = {}, {}
    fan_in = D_in
    for k in range(NUM_LAYERS):
        W[k] = init_fn(fan_in, Dh, rng)
        b[k] = np.zeros(Dh)
        fan_in = Dh
    return {"W": W, "b": b}

def broken_init(fan_in, fan_out, rng):
    """THE BUGGY INITIALIZER: fixed variance 1.0, ignoring layer width. Do not edit."""
    variance = 1.0
    return rng.standard_normal((fan_out, fan_in)) * np.sqrt(variance)

def layer_variances(X, params):
    """Given to you. Runs the forward pass and returns the list
    [Var(F[0]), Var(F[1]), ..., Var(F[last])] — the pre-activation variance per layer."""
    _, cache = forward(X, params)
    return [float(np.var(Fk)) for Fk in cache["F"]]

# ============================================================================
# TODO 1 — He initialization (equation 7.32).
# Return a weight matrix of shape (fan_out, fan_in) whose entries are drawn from
# a normal distribution with mean 0 and variance 2 / fan_in.
# ============================================================================
def he_init(fan_in, fan_out, rng):
    return rng.normal(loc=0.0, scale=np.sqrt((2/fan_in)), size=(fan_out, fan_in))

# ============================================================================
# TODO 2 — diagnose the broken network's regime (equation 7.31).
# Set this to "exploding" or "vanishing".
# ============================================================================
broken_regime = "exploding"

