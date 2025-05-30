import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the edge from node i to j is 'bad' by the amount of distance it requires
    # and 'good' by how little distance it requires.
    # Since a distance matrix is symmetric, we will use only one of its triangles (e.g., upper triangle).
    # This avoids computing the same heuristics twice for undirected edges.
    
    # Calculate the minimum distance from each node to all other nodes.
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a new matrix with the same shape as the distance matrix.
    heuristics = np.zeros_like(distance_matrix)
    
    # Fill the matrix with the heuristics. For the diagonal, set the value to 0 as it doesn't apply.
    np.fill_diagonal(heuristics, 0)
    
    # Copy the min distance to each edge, since we assume shorter distances are preferable.
    heuristics = np.copy(min_distances)
    
    return heuristics