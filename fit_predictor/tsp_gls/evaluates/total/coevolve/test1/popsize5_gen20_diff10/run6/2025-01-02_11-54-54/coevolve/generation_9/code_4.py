import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance for each row
    min_row_distances = np.min(distance_matrix, axis=1)
    
    # Subtract the minimum distance from each element in the matrix
    normalized_matrix = distance_matrix - min_row_distances
    
    # Calculate the total cost of the graph
    total_cost = np.sum(distance_matrix)
    
    # Normalize the matrix by dividing each element by the total cost
    normalized_matrix /= total_cost
    
    # The heuristics are the normalized matrix, which now represents the relative "badness"
    return normalized_matrix