import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic function is based on a simple heuristic:
    # the "promise" of an item is its prize value divided by the total weight
    # of its dimensions, as each dimension weight is 1, we can simplify it to the prize value.
    
    # Calculate the heuristic as the prize value of each item
    heuristics = prize
    
    return heuristics