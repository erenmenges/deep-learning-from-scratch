import numpy as np

def hetero_loss_BUGGY(y, f1, f2):
    """
    Heteroscedastic regression NLL (BROKEN).
    y, f1, f2 : shape (I,)
      f1 = predicted mean  mu_i        (eq 5.14)
      f2 = 2nd network output; predicted variance is sigma2_i = f2_i**2  (eq 5.14)
    Returns a scalar loss.
    """
    sigma2 = f2**2
    return np.sum((y - f1)**2 / (2.0 * sigma2))

# it's wrong because it doesn't have the inverse scaling term. normally that would be ignored because sigma2 would be a constant but now it can't 
# be ignored and it has to be there in the loss function. also the minus is not there.

def hetero_loss(y, f1, f2):
    """
    Heteroscedastic regression NLL .
    y, f1, f2 : shape (I,)
      f1 = predicted mean  mu_i        (eq 5.14)
      f2 = 2nd network output; predicted variance is sigma2_i = f2_i**2  (eq 5.14)
    Returns a scalar loss.
    """
    sigma2 = f2**2
    return -1 * np.sum(np.log(1 / (np.sqrt(2 * np.pi * sigma2))) - ((y - f1)**2 / (2.0 * sigma2)))

