import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic is simply the prize per unit weight,
    # and that the constraint is fixed to 1 for each dimension.
    # The heuristic is calculated as the sum of prizes divided by the sum of weights.
    # In reality, this should be replaced with a more complex heuristic.
    total_prize = np.sum(prize)
    total_weight = np.sum(weight, axis=1)
    
    # To avoid division by zero, we add a small epsilon to the total_weight.
    epsilon = 1e-9
    total_weight += epsilon
    
    # Calculate the heuristic score as the prize per unit weight.
    heuristics = total_prize / total_weight
    
    return heuristics

# Example usage:
# prize = np.array([100, 200, 300])
# weight = np.array([[1, 2], [2, 1], [3, 1]])
# print(heuristics_v2(prize, weight))