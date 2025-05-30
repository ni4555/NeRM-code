import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized prize per unit weight for each item and dimension
    normalized_prize = prize / weight
    
    # Calculate the sum of normalized prizes for each item
    item_promise = np.sum(normalized_prize, axis=1)
    
    # Return the item promise as the heuristic
    return item_promise