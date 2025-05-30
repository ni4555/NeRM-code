import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Iterate over each item
    for i in range(prize.shape[0]):
        # Calculate the total weight of the item across all dimensions
        total_weight = np.sum(weight[i])
        
        # Calculate the heuristic based on prize and total weight
        # This is a simple heuristic: the higher the prize, the more promising the item
        heuristics[i] = prize[i] / total_weight
    
    # Normalize the heuristics so that they sum to 1
    heuristics /= np.sum(heuristics)
    
    return heuristics