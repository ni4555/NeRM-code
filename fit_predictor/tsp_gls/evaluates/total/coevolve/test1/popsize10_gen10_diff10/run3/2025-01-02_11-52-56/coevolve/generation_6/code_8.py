import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric and the diagonal elements are 0
    # Calculate the heuristic for each edge
    # Here we use a simple heuristic based on the average distance to the nearest neighbor
    # This is just an example heuristic, more sophisticated ones could be implemented
    
    # Calculate the average distance to the nearest neighbor for each city
    avg_distances = np.array([np.mean(distance_matrix[i, :i] + distance_matrix[i, i+1:])
                              for i in range(len(distance_matrix))])
    
    # Create a matrix where each element is the average distance of the corresponding edge
    # multiplied by a factor to ensure non-negative values (the factor can be adjusted)
    factor = 1.1  # This factor can be tuned
    heuristic_matrix = avg_distances * factor
    
    return heuristic_matrix