import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize a heuristic value array with zeros
    heuristics = np.zeros_like(prize)
    
    # Compute the heuristics based on the prize-to-weight ratio
    for i in range(len(prize)):
        # Only if the weight for dimension 0 is 1, as per the constraint
        if weight[i][0] == 1:
            heuristics[i] = prize[i] / weight[i][0]
    
    return heuristics