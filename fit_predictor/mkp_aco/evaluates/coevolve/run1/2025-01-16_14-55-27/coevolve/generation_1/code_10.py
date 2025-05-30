import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the problem is a multi-dimensional knapsack problem with each item's weight
    # having a fixed dimension constraint of 1, we can use the ratio of prize to weight as
    # a heuristic. This is a common heuristic for knapsack problems where the goal is to maximize
    # the total prize collected.

    # Calculate the prize-to-weight ratio for each item
    prize_to_weight_ratio = prize / weight.sum(axis=1)

    # Sort the items based on the prize-to-weight ratio in descending order
    sorted_indices = np.argsort(-prize_to_weight_ratio)

    # Return the sorted indices as the heuristic scores
    return sorted_indices