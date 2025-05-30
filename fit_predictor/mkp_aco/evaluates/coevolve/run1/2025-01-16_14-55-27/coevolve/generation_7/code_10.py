import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)

    # Assuming that the prize and weight arrays have the same length 'n'
    n = prize.size

    # Calculate the normalized prize value for each item
    normalized_prize = prize / np.sum(prize)

    # Calculate the normalized weight for each item using the dynamic weighted ratio index
    # Since the constraint for each dimension is fixed to 1 and 'm' is the dimension,
    # we can simply take the average of the weights across the dimensions
    normalized_weight = np.mean(weight, axis=1)

    # Compute the heuristic value for each item
    # Heuristic is calculated as the normalized prize divided by the normalized weight
    heuristics = normalized_prize / normalized_weight

    return heuristics