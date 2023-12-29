import numpy as np


# Example distributions
p = np.array([0.4, 0.6])
q = np.array([0.35, 0.65])


def kl_divergence(p, q):
    """Compute the KL divergence of q from p (D_KL(p||q))."""
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    """Compute the Jensen-Shannon divergence between p and q."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

print(hellinger_distance(p, q))
# Compute KL divergence
divergence = kl_divergence(p, q)
print(f"KL Divergence (D_KL(p||q)): {divergence:.4f}")

# Compute JS divergence
js_div = js_divergence(p, q)
print(f"JS Divergence: {js_div:.4f}")
