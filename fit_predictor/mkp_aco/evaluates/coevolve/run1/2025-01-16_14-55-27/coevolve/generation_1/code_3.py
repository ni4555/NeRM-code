import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic function will calculate the importance of each item
    # based on some criteria. For the sake of demonstration, let's use a simple heuristic
    # which calculates the ratio of prize to weight for each item.
    # In a real-world scenario, you would replace this with a more sophisticated heuristic.
    
    # Calculate the heuristic value for each item
    heuristic_values = prize / weight.sum(axis=1)
    
    # Normalize the heuristic values to ensure they sum to 1
    heuristic_values /= heuristic_values.sum()
    
    return heuristic_values