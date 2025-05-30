import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    heuristic = np.zeros(n)
    
    # Sample the items based on their probability to be included in the knapsacks
    for item in range(n):
        probability = prize[item] / (np.sum(weight[item] * np.exp(prize)))
        if np.random.rand() < probability:
            heuristic[item] = 1
            
    return heuristic
