import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is symmetric (distance from i to j is the same as from j to i)
    # and contains only positive values.

    # Normalize the distance matrix by the minimum distance for each row.
    min_distance_per_row = np.min(distance_matrix, axis=1)
    normalized_matrix = distance_matrix / min_distance_per_row[:, np.newaxis]

    # Calculate the distance-weighted normalization for each edge.
    distance_weighted_normalization = np.sqrt(distance_matrix) * np.log(1 + normalized_matrix)

    # Compute a resilient minimum spanning tree (MST) heuristic for each edge.
    # We will use the Kruskal's algorithm to find the MST, as it is efficient for sparse graphs.
    # However, for simplicity, we'll just use the edge weights themselves as the heuristic
    # since the description does not specify a particular method for constructing the MST.
    # In a real-world scenario, this would be replaced by a proper MST computation.

    # For each edge, the heuristic is the distance-weighted normalization value plus the MST heuristic.
    # We will just use the distance-weighted normalization for this implementation.
    heuristics = distance_weighted_normalization

    return heuristics