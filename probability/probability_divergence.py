import numpy as np

def kl_divergence(p, q):
    """Compute the KL divergence of q from p (D_KL(p||q))."""
    return np.sum(p * np.log(p / q))

# Example distributions
p = np.array([0.4, 0.6])
q = np.array([0.35, 0.65])

# Compute KL divergence
divergence = kl_divergence(p, q)
print(f"KL Divergence (D_KL(p||q)): {divergence:.4f}")



def js_divergence(p, q):
    """Compute the Jensen-Shannon divergence between p and q."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Compute JS divergence
js_div = js_divergence(p, q)
print(f"JS Divergence: {js_div:.4f}")
