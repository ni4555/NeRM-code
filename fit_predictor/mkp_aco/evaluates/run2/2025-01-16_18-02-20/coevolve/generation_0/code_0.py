import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the inverse of the weighted sum of prizes to normalize the heuristics
    weighted_prizes = np.sum(prize * weight, axis=1)
    normalized_weights = 1 / (weighted_prizes + 1e-8)  # Adding a small constant to avoid division by zero
    # Normalize by the maximum value to ensure non-negative heuristics
    max_normalized_weight = np.max(normalized_weights)
    heuristics = normalized_weights / max_normalized_weight
    return heuristics