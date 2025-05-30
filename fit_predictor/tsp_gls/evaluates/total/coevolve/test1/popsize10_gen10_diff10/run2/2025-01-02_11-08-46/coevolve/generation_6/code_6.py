import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix to account for distance-based normalization
    distance_matrix = np.max(distance_matrix, axis=1) / distance_matrix
    
    # Apply the robust minimum sum heuristic for optimal edge selection
    # This step might require a custom heuristic function or a simplification
    # since the actual implementation details of the heuristic are not specified
    # Here we will assume a simple heuristic for demonstration purposes:
    # We will calculate the minimum sum of distances for each node, then
    # normalize the distances relative to this sum
    min_sums = np.sum(distance_matrix, axis=0)
    min_sums = np.where(min_sums == 0, 1, min_sums)  # Avoid division by zero
    normalized_distances = distance_matrix / min_sums[:, np.newaxis]
    
    # The output is of the same shape as the input, and it represents the heuristics
    # which is a measure of how "bad" it is to include each edge
    return normalized_distances