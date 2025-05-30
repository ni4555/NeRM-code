import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize to be between 0 and 1
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the heuristic based on normalized prize
    heuristics = normalized_prize * np.sum(weight, axis=1)
    
    return heuristics
