import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Calculate the utility of each item as the normalized prize value
    utility = prize / np.sum(prize)
    
    # Calculate the load of each item in each dimension
    load = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Initialize a matrix to store the heuristic value for each item in each dimension
    heuristic_matrix = np.zeros((n, m))
    
    # Calculate the heuristic for each item in each dimension
    for i in range(n):
        for j in range(m):
            # Calculate the heuristic value based on the load and utility
            heuristic_matrix[i, j] = load[i, j] * utility[i]
    
    # Sum the heuristic values across dimensions to get the final heuristic for each item
    final_heuristic = np.sum(heuristic_matrix, axis=1)
    
    # Apply a selection heuristic to prioritize diverse solutions
    selected_indices = np.argsort(final_heuristic)[-n//2:]  # Select top n//2 items for diversity
    final_heuristic[selected_indices] = 1.5 * final_heuristic[selected_indices]  # Increase their heuristic value
    
    return final_heuristic
