import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Euclidean distance heuristics
    euclidean_heuristics = np.sqrt(np.sum(distance_matrix**2, axis=1))
    
    # Calculate Chebyshev distance heuristics
    chebyshev_heuristics = np.max(distance_matrix, axis=1)
    
    # Combine both heuristics, using Chebyshev distance as a base
    combined_heuristics = chebyshev_heuristics + 0.5 * (euclidean_heuristics - chebyshev_heuristics)
    
    return combined_heuristics