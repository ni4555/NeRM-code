import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Step 1: Calculate the value-to-weight ratio for each item and each dimension
    value_to_weight_ratio = np.dot(prize, np.ones(m)) / np.dot(weight, np.ones(m))
    
    # Step 2: Sort items by their total value-to-weight ratio
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Step 3: Calculate the total weight of each item based on sorted order
    cumulative_weight = np.cumsum(weight[sorted_indices])
    
    # Step 4: Apply a greedy approach to select items based on the cumulative weight constraint
    selected_indices = np.where(cumulative_weight <= 1)[0]
    
    # Step 5: Calculate the heuristic score for each item
    heuristics = np.zeros(n)
    heuristics[selected_indices] = 1.0
    
    # Step 6: Normalize the heuristic scores
    # Normalize based on the fraction of the total weight that each item contributes
    normalized_heuristics = heuristics / np.sum(heuristics)
    
    return normalized_heuristics