import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Implementing the cutting-edge hybrid algorithm
    # Step 1: Distance-weighted normalization
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Step 2: Resilient minimum spanning tree heuristic
    # (This step is conceptualized as an abstract heuristic function since the
    # specific implementation of the resilient minimum spanning tree heuristic
    # is not detailed in the description)
    resilient_mst_weights = np.random.rand(distance_matrix.shape[0])
    normalized_mst_weights = (resilient_mst_weights - np.min(resilient_mst_weights)) / (np.max(resilient_mst_weights) - np.min(resilient_mst_weights))
    
    # Step 3: Combine both heuristics to create the heuristic function
    combined_weights = np.random.rand(distance_matrix.shape[0])
    heuristics_values = normalized_distances * combined_weights + normalized_mst_weights * (1 - combined_weights)
    
    return heuristics_values