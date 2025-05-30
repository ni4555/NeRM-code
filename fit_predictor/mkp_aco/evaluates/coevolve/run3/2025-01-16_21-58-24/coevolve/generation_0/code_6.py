import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is to calculate the prize per unit weight for each item
    # and then sort by this value to get the most promising items.
    # The heuristic assumes that the constraint of each dimension is fixed to 1,
    # so we can simply use the sum of weights along each dimension (which will be 1 for all items).
    
    # Calculate the prize per unit weight for each item
    prize_per_unit_weight = prize / weight.sum(axis=1)
    
    # Sort items by the prize per unit weight in descending order
    sorted_indices = np.argsort(prize_per_unit_weight)[::-1]
    
    # Create a boolean array to indicate if the item is promising based on the heuristic
    promising = np.zeros_like(prize, dtype=bool)
    promising[sorted_indices] = True
    
    return promising