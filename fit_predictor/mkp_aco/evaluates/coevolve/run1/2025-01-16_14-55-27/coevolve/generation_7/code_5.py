import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = len(prize)
    m = len(weight[0])
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros(n)
    
    # Calculate the weighted ratio index for each item
    for i in range(n):
        dynamic_weighted_ratio = np.sum(prize[i] / weight[i])
        heuristics[i] = dynamic_weighted_ratio
    
    # Normalize the heuristics based on the maximum value
    max_heuristic = np.max(heuristics)
    heuristics = heuristics / max_heuristic
    
    return heuristics