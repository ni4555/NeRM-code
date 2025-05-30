import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Normalize prize for each dimension
    normalized_prizes = prize / np.sum(prize, axis=0)
    # Calculate value-to-weight ratio for each item
    value_to_weight = normalized_prizes / weight
    # Calculate the heuristic score as the sum of the value-to-weight ratios
    heuristics = np.sum(value_to_weight, axis=1)
    # Normalize the heuristic scores to make them more interpretable
    heuristics = (heuristics - np.min(heuristics)) / (np.max(heuristics) - np.min(heuristics))
    return heuristics