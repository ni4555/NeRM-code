import numpy as np
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is symmetric
    distance_matrix = np.tril(distance_matrix) + np.tril(distance_matrix, -1).T
    
    # Create a sparse representation of the distance matrix
    distance_sparse = csr_matrix(distance_matrix)
    
    # Compute the Minimum Spanning Tree (MST)
    mst_edges = minimum_spanning_tree(distance_sparse)
    
    # Extract the edge weights from the MST
    mst_weights = mst_edges.data
    
    # Adjust the MST weights based on the current population diversity
    # Assuming 'population_diversity' is a measure of diversity that affects the heuristic
    # For simplicity, we use the standard deviation of the edge weights as a proxy for diversity
    population_diversity = np.std(mst_weights)
    
    # Adjust the heuristic by adding a term proportional to the diversity
    adjusted_weights = mst_weights + 0.01 * population_diversity
    
    # Convert the adjusted weights back to a numpy array
    adjusted_weights_array = np.array(adjusted_weights).reshape(distance_matrix.shape)
    
    return adjusted_weights_array