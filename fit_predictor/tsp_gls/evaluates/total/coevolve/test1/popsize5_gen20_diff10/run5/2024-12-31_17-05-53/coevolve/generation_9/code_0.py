import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The heuristic matrix will initially be a copy of the distance matrix.
    # The heuristic for each edge will be the inverse of its distance, which
    # is a common approach in TSP heuristics.
    heuristic_matrix = np.copy(distance_matrix)
    
    # We'll use the Manhattan distance as a heuristic for the heuristic matrix.
    # The Manhattan distance for an edge from node i to node j is the sum of the
    # distances from i to each node in the row and from j to each node in the column
    # that are not i or j.
    for i in range(heuristic_matrix.shape[0]):
        for j in range(heuristic_matrix.shape[1]):
            if i != j:
                # Sum the distances for the Manhattan heuristic
                Manhattan_heuristic = np.sum(np.abs(heuristic_matrix[i] - heuristic_matrix[j])) - distance_matrix[i][j]
                # Assign the Manhattan heuristic to the edge (i, j)
                heuristic_matrix[i][j] = Manhattan_heuristic
    
    # We want to make sure that the diagonal elements (self-loops) are set to a high
    # value, as they should not be included in the solution.
    np.fill_diagonal(heuristic_matrix, np.inf)
    
    return heuristic_matrix