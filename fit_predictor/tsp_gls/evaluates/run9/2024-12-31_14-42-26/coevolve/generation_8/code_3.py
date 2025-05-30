import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Placeholder for an advanced heuristic computation
    # This is a simple example where we use a normalized distance for the heuristic value.
    # The actual heuristic should be more complex and intelligent as described in the problem statement.
    
    # Calculate the heuristic values based on a normalization of the distances
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Add some random noise to simulate exploration (this could be replaced with more sophisticated logic)
    noise = np.random.rand(*distance_matrix.shape) * 0.1
    heuristics = normalized_distances + noise
    
    return heuristics