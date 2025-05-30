import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance as a heuristic for each edge
    # This is done by considering the sum of the absolute differences in the coordinates
    # between every pair of cities.
    num_cities = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):  # We only calculate the distance once for each edge
            # Assuming that the distance matrix is 2D with city coordinates
            # Extract the coordinates of city i and city j
            city_i_coords = np.array([i // num_cities, i % num_cities])
            city_j_coords = np.array([j // num_cities, j % num_cities])
            
            # Calculate the Manhattan distance
            manhattan_distance = np.sum(np.abs(city_i_coords - city_j_coords))
            
            # Set the heuristic for the edge (i, j)
            heuristics[i, j] = heuristics[j, i] = manhattan_distance
    
    return heuristics