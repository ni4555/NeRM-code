import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the sum of weights for each item across all dimensions
    item_weight_sums = np.sum(weight, axis=1)
    # Calculate the sum of prizes for each item across all dimensions
    item_prize_sums = np.sum(prize, axis=1)
    # Calculate the heuristic value for each item as the ratio of prize to weight sum
    heuristics = item_prize_sums / item_weight_sums
    return heuristics