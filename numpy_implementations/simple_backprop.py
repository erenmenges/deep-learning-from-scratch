import numpy as np

def relu_forward(f: np.ndarray) -> np.ndarray:
    return np.maximum(f, 0)

def relu_backward(grad_h: np.ndarray, f: np.ndarray) -> np.ndarray:
    indicator = (f > 0)  ### bool array, maybe use astype(float) next time
    return grad_h * indicator


def toy_forward(x: float, params: dict) -> tuple[float, dict]:
    intermediates = {}
    f0 = params["b0"] + params["w0"] * x
    intermediates["f0"] = f0
    h1 = np.sin(f0)
    intermediates["h1"] = h1
    f1 = params["b1"] + params["w1"] * h1
    intermediates["f1"] = f1
    h2 = np.exp(f1)
    intermediates["h2"] = h2
    f2 = params["b2"] + params["w2"] * h2
    intermediates["f2"] = f2
    h3 = np.cos(f2)
    intermediates["h3"] = h3
    f3 = params["b3"] + params["w3"] * h3
    intermediates["f3"] = f3
    return (f3, intermediates)

def toy_grads(x: float, y: float, params: dict) -> dict:
    f3, intermediates = toy_forward(x, params)
    derivatives = {}
    derivatives["li/f3"] = 2 * (f3 - y)
    derivatives["li/h3"] = params["w3"] * derivatives["li/f3"]
    derivatives["li/f2"] = -1 * np.sin(intermediates["f2"]) * derivatives["li/h3"]
    derivatives["li/h2"] = params["w2"] * derivatives["li/f2"]
    derivatives["li/f1"] = np.exp(intermediates["f1"]) * derivatives["li/h2"]
    derivatives["li/h1"] = params["w1"] * derivatives["li/f1"]
    derivatives["li/f0"] = np.cos(intermediates["f0"]) * derivatives["li/h1"]

    param_derivatives = {}
    param_derivatives["w3"] = intermediates["h3"] * derivatives["li/f3"]
    param_derivatives["w2"] = intermediates["h2"] * derivatives["li/f2"]
    param_derivatives["w1"] = intermediates["h1"] * derivatives["li/f1"]
    param_derivatives["w0"] = x * derivatives["li/f0"]

    param_derivatives["b3"] = 1 * derivatives["li/f3"]
    param_derivatives["b2"] = 1 * derivatives["li/f2"]
    param_derivatives["b1"] = 1 * derivatives["li/f1"]
    param_derivatives["b0"] = 1 * derivatives["li/f0"]
    return param_derivatives

