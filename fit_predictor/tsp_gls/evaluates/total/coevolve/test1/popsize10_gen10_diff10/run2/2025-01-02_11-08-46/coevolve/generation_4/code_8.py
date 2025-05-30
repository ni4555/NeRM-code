import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Step 1: Distance-weighted normalization
    # Normalize distances so that the maximum distance is 1, keeping relative distances
    max_distance = np.max(distance_matrix)
    distance_normalized = distance_matrix / max_distance
    
    # Step 2: Resilient minimum spanning tree (MST) heuristic
    # Initialize the MST with a single vertex
    num_vertices = distance_matrix.shape[0]
    selected = [False] * num_vertices
    selected[0] = True
    mst_edges = []
    edge_weights = np.inf
    
    # Construct the MST using Prim's algorithm
    while True:
        next_min_edge = None
        for i in range(num_vertices):
            if selected[i]:
                for j in range(num_vertices):
                    if not selected[j] and distance_normalized[i, j] < edge_weights:
                        next_min_edge = (i, j)
                        edge_weights = distance_normalized[i, j]
        
        if next_min_edge is None:
            break
        
        # Add the minimum edge to the MST
        i, j = next_min_edge
        mst_edges.append((i, j))
        selected[j] = True
        edge_weights = np.inf
    
    # Step 3: Combine the distance-weighted normalization and the MST heuristic
    # Create a boolean mask for edges in the MST
    mst_mask = np.zeros_like(distance_normalized)
    for i, j in mst_edges:
        mst_mask[i, j] = mst_mask[j, i] = True
    
    # Weigh the MST edges less than non-MST edges
    heuristic_values = distance_normalized.copy()
    for i in range(num_vertices):
        for j in range(num_vertices):
            if not mst_mask[i, j]:
                # Apply a larger penalty to non-MST edges
                heuristic_values[i, j] *= 1.5
    
    return heuristic_values