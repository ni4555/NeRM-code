import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Assuming that the heuristic function will use some heuristic algorithm
    # For example, here we are using a simple heuristic that assumes higher prize
    # items are more promising. This is just a placeholder for a real heuristic.
    heuristics = prize / np.sum(weight, axis=1)
    
    return heuristics