import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio metric
    weighted_prize = np.sum(prize * weight, axis=1)
    # Apply cumulative sum analysis for precise item contribution assessment
    cumulative_sum = np.cumsum(weight, axis=1)
    # Combine the weighted prize with the cumulative sum for a multi-dimensional weighted ratio
    heuristics = weighted_prize / cumulative_sum
    return heuristics