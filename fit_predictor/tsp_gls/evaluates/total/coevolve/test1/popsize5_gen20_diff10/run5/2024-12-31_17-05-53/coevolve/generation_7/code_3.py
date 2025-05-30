import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance for each edge in the distance matrix
    # The Manhattan distance between two points (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|
    # We use this heuristic to estimate the "badness" of an edge
    # The result is a matrix of the same shape as the input distance matrix
    heuristics_matrix = np.abs(distance_matrix - np.tril(distance_matrix, k=-1))
    heuristics_matrix = np.abs(heuristics_matrix - np.triu(distance_matrix, k=1))
    
    # Normalize the heuristics matrix to ensure that all values are positive and sum to 1
    heuristics_matrix = np.maximum(heuristics_matrix, 0)
    heuristics_matrix /= np.sum(heuristics_matrix, axis=1, keepdims=True)
    
    return heuristics_matrix