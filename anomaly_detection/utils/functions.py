import numpy as np


def compute_mahalanobis_scores(
    X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray
) -> np.ndarray:
    """Compute Mahalanobis distances for input data."""
    if X.size == 0:
        return np.array([])

    # Center the data
    centered = X - mean[None, :]

    # Compute quadratic form: (x - μ)ᵀ Σ⁻¹ (x - μ)
    left = centered.dot(inv_cov)
    scores = np.einsum("ij,ij->i", left, centered)

    return scores
