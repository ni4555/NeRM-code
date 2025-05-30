import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Implementing a simple heuristic based on distance-based normalization and minimum sum heuristic
    # For simplicity, let's assume the distance matrix is symmetric (i.e., distance from A to B is the same as from B to A)
    # and that all distances are greater than zero.
    
    # Distance-based normalization: Divide each distance by the sum of all distances from the same node.
    normalized_distances = distance_matrix / np.sum(distance_matrix, axis=1, keepdims=True)
    
    # Robust minimum sum heuristic: Find the minimum sum of distances for each node as a potential edge selection.
    min_sum_heuristic = np.min(distance_matrix, axis=0)
    
    # Combine the heuristics by taking the weighted sum where the weight is inversely proportional to the normalized distance.
    # This is a simple way to incorporate the enhanced heuristic function.
    heuristic_weights = 1 / (normalized_distances + 1e-8)  # Adding a small constant to avoid division by zero
    combined_heuristic = heuristic_weights * min_sum_heuristic
    
    return combined_heuristic