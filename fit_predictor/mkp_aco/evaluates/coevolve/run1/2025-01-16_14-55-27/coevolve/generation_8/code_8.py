import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure that the input dimensions are consistent
    if prize.shape[0] != weight.shape[0]:
        raise ValueError("The number of items in prize and weight must be the same.")

    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)

    # Calculate the weighted ratio for each item
    # Since each dimension's constraint is 1, the sum of weights across dimensions gives the total weight
    weighted_ratio = np.sum(prize, axis=1) / total_weight

    # Sort items based on the weighted ratio in descending order
    sorted_indices = np.argsort(weighted_ratio)[::-1]

    # Calculate the heuristics score by normalizing the sorted indices
    # Higher index in the sorted order means better heuristic
    heuristics = np.arange(weight.shape[0])[sorted_indices]

    return heuristics