import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric and the diagonal elements are zeros
    # We will calculate the heuristic values for each edge based on some criteria
    # Here we will use a simple heuristic that assumes the cost of an edge is inversely proportional to its distance
    # For edges with distance 0 (diagonal elements), we assign a high cost to avoid including them in the solution
    heuristic_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero
    return heuristic_matrix