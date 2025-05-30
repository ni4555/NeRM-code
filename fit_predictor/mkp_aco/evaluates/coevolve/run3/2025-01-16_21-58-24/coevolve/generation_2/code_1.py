import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic: use the ratio of prize to total weight in each dimension
    total_weight_per_dimension = np.sum(weight, axis=0)
    # Normalize by the total weight for each item to make it comparable
    normalized_weights = weight / total_weight_per_dimension[:, np.newaxis]
    # Calculate the heuristic based on the normalized prize-weight ratio
    heuristics = prize / np.maximum(1e-8, normalized_weights)  # Add a small epsilon to avoid division by zero
    return heuristics