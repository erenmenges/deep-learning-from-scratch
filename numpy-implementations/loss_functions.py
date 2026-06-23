import numpy as np

def normal_pdf(y: np.ndarray, mu: np.ndarray, sigma2: float) -> np.ndarray:
    return ((1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-1 * (((y - mu) ** 2))/(2 * sigma2)))

def regression_nll(y: np.ndarray, mu: np.ndarray, sigma2: float) -> float:
    return -1 * np.sum(np.log(normal_pdf(y, mu, sigma2)))

def least_squares_loss(y: np.ndarray, mu: np.ndarray) -> float:
    return np.sum((y - mu)**2)

def sigmoid(f: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-1 * f)))

def bce_loss(f: np.ndarray, y: np.ndarray) -> float:
    return np.sum(((y - 1) * np.log(1 - sigmoid(f))) - (y * np.log(sigmoid(f))))

def bce_predict(f: np.ndarray) -> np.ndarray:
    return np.where(sigmoid(f) > 0.5, 1, 0)

def softmax(F: np.ndarray):
    return np.exp(F) / np.sum(np.exp(F), axis=1)[:, None]

def multiclass_ce_loss(F: np.ndarray, y: np.ndarray) -> float:
    correct_logits = F[np.arange(F.shape[0]), y]
    return -1 * np.sum(correct_logits - np.log(np.sum(np.exp(F), axis=1)))

def multiclass_predict(F: np.ndarray) -> np.ndarray:
    return np.argmax(softmax(F), axis=1)
