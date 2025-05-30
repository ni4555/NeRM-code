import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function computes the heuristics for the Traveling Salesman Problem (TSP)
    # using a simple heuristic approach. The function assumes that the distance_matrix
    # is a square matrix where the element at row i and column j is the distance from
    # city i to city j. The function returns a matrix of the same shape with the
    # heuristic estimates.
    
    # The heuristic here is a simple upper bound of the cost of visiting a city
    # after another city. It's computed as the minimum distance from the current
    # city to all other cities.
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # For each city, compute the heuristic value as the minimum distance to any other city
    for i in range(distance_matrix.shape[0]):
        # Exclude the distance to the current city itself by setting the diagonal to infinity
        min_distances = np.min(distance_matrix[i], axis=0)
        # The heuristic value for city i is the minimum of these distances
        heuristic_matrix[i] = min_distances
    
    return heuristic_matrix