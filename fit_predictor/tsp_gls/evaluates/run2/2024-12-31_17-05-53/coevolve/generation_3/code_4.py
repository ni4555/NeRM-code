import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the sum of distances to each vertex
    sum_distances = np.sum(distance_matrix, axis=1)
    
    # The heuristic score for each edge is the sum of distances minus the distance of the edge itself
    # (assuming the distance matrix is symmetric, the diagonal is zero)
    heuristic_scores = sum_distances - distance_matrix.diagonal()
    
    # To ensure the same shape as the input, we pad the resulting array with zeros
    padded_scores = np.pad(heuristic_scores, ((0, 0), (1, 1)), 'constant', constant_values=(0, 0))
    
    # We need to account for the fact that we cannot include an edge that loops back to the same vertex
    # So we set the diagonal of the heuristic matrix to infinity (or a very high number)
    padded_scores += np.abs(distance_matrix - np.diag(np.ones(distance_matrix.shape[0])))
    
    return padded_scores