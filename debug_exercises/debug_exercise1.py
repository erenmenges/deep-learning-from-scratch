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


import numpy as np, torch

rng = np.random.default_rng(2)
I  = 40
y  = rng.normal(size=I)
f1 = y + rng.normal(scale=0.3, size=I)     # decent mean predictions (same in both configs)
f2_honest   = np.full(I, 0.5)              # std ~ 0.5
f2_inflated = np.full(I, 50.0)             # absurdly large std, SAME means f1

def _torch_gauss_nll(y, f1, f2):
    d = torch.distributions.Normal(torch.tensor(f1), torch.tensor(np.abs(f2)))
    return -d.log_prob(torch.tensor(y)).sum().item()

# (a) corrected loss == Gaussian NLL with std |f2|  (eq 5.15 with eq 5.14)
assert np.isclose(hetero_loss(y, f1, f2_honest),   _torch_gauss_nll(y, f1, f2_honest),   atol=1e-6), "hetero_loss wrong (5.15)"
assert np.isclose(hetero_loss(y, f1, f2_inflated), _torch_gauss_nll(y, f1, f2_inflated), atol=1e-6), "hetero_loss wrong (5.15)"

# (b) the pathology: BUGGY loss is fooled (lower for inflated variance);
#     corrected loss penalizes inflation (higher for inflated variance)
assert hetero_loss_BUGGY(y, f1, f2_inflated) < hetero_loss_BUGGY(y, f1, f2_honest), \
    "buggy loss should be MINIMIZED by inflating variance"
assert hetero_loss(y, f1, f2_inflated) > hetero_loss(y, f1, f2_honest), \
    "corrected loss should PENALIZE inflated variance"

# (c) the two losses differ by exactly the dropped normalizer  sum( 0.5*log(2*pi*f2^2) )
dropped = np.sum(0.5 * np.log(2*np.pi * f2_honest**2))
assert np.isclose(hetero_loss(y, f1, f2_honest) - hetero_loss_BUGGY(y, f1, f2_honest), dropped, atol=1e-8), \
    "the only difference should be the Gaussian normalizer term"
print("Problem 3: all checks passed.")