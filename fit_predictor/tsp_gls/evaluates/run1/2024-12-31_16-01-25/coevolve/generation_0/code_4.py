import numpy as np
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Create a symmetric distance matrix if it is not already
    if np.any(distance_matrix != distance_matrix.T):
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    
    # Compute the minimum spanning tree (MST) of the distance matrix
    # The minimum spanning tree will give us the "best" edges to include in our heuristic
    mst = minimum_spanning_tree(distance_matrix)
    
    # The `minimum_spanning_tree` function returns a sparse matrix of the MST
    # We convert it to a dense matrix for the return value
    mst_dense = mst.toarray()
    
    # We want to return a matrix where each entry is the heuristic value for the edge
    # Since the MST does not have any self-loops or repeated edges, we can simply return the
    # MST matrix itself as the heuristic. The heuristic value for each edge is 0, and for the
    # MST edges, it is the weight of the edge.
    return mst_dense