import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure the weights are all 1s
    if np.any(weight != 1):
        raise ValueError("Each item's weight should be 1, but got:", weight)
    
    # Calculate the "promise" for each item as the prize value divided by the weight
    # In this case, since weight is always 1, the promise is just the prize value
    promise = prize / weight
    
    # Return the promises (promising values) as an array
    return promise

# Example usage:
# n = 5
# m = 1
# prize = np.array([10, 20, 30, 40, 50])
# weight = np.array([[1], [1], [1], [1], [1]])
# print(heuristics_v2(prize, weight))