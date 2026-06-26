import numpy as np

rng = np.random.default_rng(0)
N = 50
x = rng.uniform(-5.0, 5.0, size=N) * 50.0      # large-scale feature -> anisotropic loss
true_phi = np.array([2.0, 0.3])
y = true_phi[0] + true_phi[1]*x + rng.normal(0.0, 1.0, size=N)

def model(phi, x): return phi[0] + phi[1]*x
def loss(phi):     r = model(phi, x) - y; return float(np.sum(r**2))
def grad(phi):     r = model(phi, x) - y; return np.array([2*np.sum(r), 2*np.sum(r*x)])

# ---- BROKEN training loop: fixed-step GD on an anisotropic surface ----
"""
phi = np.array([0.0, 0.0])
alpha = 0.001
for t in range(200):
    phi = phi - alpha * grad(phi)
print(loss(phi), phi)
"""
# TODO: diagnose (comment) and replace this loop with a converging in-scope optimizer.

# comment: it doesn't work because a small alpha only is efficient if both directions are equally steep. in here a small alpha would slow down
# a lot when it has to go in the other, flat direction. a large alpha would overshoot in the steep direction, which would destabilize the loss.

# fixed using adam optimizer:

phi = np.array([0.0, 0.0])

alpha=0.05
beta=0.9
gamma=0.99
eps=1e-8
m_next = 0
v_next = 0

for t in range(1,201):
    g = grad(phi)
    m_next = (beta * m_next) + ((1 - beta) * g)
    v_next = (gamma * v_next) + ((1 - gamma) * (g**2))

    m_hat_next = m_next / (1 - (beta**t))
    v_hat_next = v_next / (1 - (gamma**t))
    phi = phi - alpha * (m_hat_next/(np.sqrt(v_hat_next) + eps))
print(loss(phi), phi)
