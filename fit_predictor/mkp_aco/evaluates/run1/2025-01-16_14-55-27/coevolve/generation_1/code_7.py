import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Since each dimension weight constraint is fixed to 1, we can sum the weights across dimensions
    # to determine the total weight of each item.
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic as the prize divided by the total weight for each item.
    # This heuristic could be interpreted as the maximum prize per unit weight that can be gained
    # from including an item in the knapsack.
    heuristics = prize / total_weight
    
    # If the total weight is zero for any item (which theoretically shouldn't happen given the constraint),
    # we avoid division by zero by setting the heuristic to a very low value.
    heuristics[np.where(total_weight == 0)] = 0
    
    return heuristics