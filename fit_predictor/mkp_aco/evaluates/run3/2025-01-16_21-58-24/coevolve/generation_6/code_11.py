import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # This is a simplified version of a heuristic function.
    # Since each dimension constraint is fixed to 1, the heuristic could be a normalized value of the profit
    # divided by the total weight of the item across all dimensions.
    
    # Calculate normalized profit per unit weight for each item
    normalized_profit = prize / np.sum(weight, axis=1)
    
    # Return the heuristic values, which can be considered as a measure of how promising an item is
    # since items with higher normalized profit are more promising to be included in the solution.
    return normalized_profit