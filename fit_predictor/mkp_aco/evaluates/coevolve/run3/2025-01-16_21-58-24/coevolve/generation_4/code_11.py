import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight of each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic score for each item based on the prize-to-weight ratio
    heuristic_scores = prize / total_weight
    
    # Normalize the heuristic scores to ensure they are non-negative
    min_score = np.min(heuristic_scores)
    heuristic_scores -= min_score
    
    # Return the heuristics array
    return heuristic_scores