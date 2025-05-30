import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic where the heuristic for each item is proportional to its prize
    # while normalized by the sum of weights across dimensions to account for varying capacities.
    total_weight = np.sum(weight, axis=1)
    # Avoid division by zero by adding a small epsilon if necessary
    epsilon = 1e-8
    total_weight = np.clip(total_weight, epsilon, None)
    normalized_prize = prize / total_weight[:, np.newaxis]
    # The heuristic value for each item is its normalized prize
    heuristics = normalized_prize
    return heuristics