import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")

    # Calculate the total distance for all edges
    total_distance = np.sum(distance_matrix)

    # Compute the average edge distance
    average_distance = total_distance / (len(distance_matrix) * (len(distance_matrix) - 1) / 2)

    # Create a heuristic matrix with the same shape as the distance matrix
    # Here we use the negative average distance to reflect a heuristic "badness"
    # since lower values indicate better solutions in minimization problems.
    heuristic_matrix = -average_distance * np.ones_like(distance_matrix)

    # Since we don't want to penalize edges that are part of the diagonal
    # (self-loops are not included in the TSP problem), we set those to 0.
    np.fill_diagonal(heuristic_matrix, 0)

    return heuristic_matrix