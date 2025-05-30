import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Step 1: Normalize the value-to-weight ratios
    value_to_weight = prize / weight
    max_ratio = np.max(value_to_weight)
    normalized_ratio = value_to_weight / max_ratio
    
    # Step 2: Calculate the probability of selection for each item
    # The probability is directly proportional to the normalized ratio
    probabilities = normalized_ratio / np.sum(normalized_ratio)
    
    # Step 3: Adaptive sampling mechanism
    # Here we use a simple heuristic where we adjust the probability
    # of selecting items based on the average weight left in the knapsacks
    # (assuming the weight limit for each knapsack is equal to the total weight of all items)
    avg_weight_left = np.sum(weight) / n
    probabilities *= avg_weight_left / weight
    
    # Normalize the probabilities again after adjustment
    probabilities /= np.sum(probabilities)
    
    # Step 4: Calculate the heuristic values for each item
    # These are the probabilities of selecting each item
    heuristics = probabilities
    
    return heuristics

# Example usage:
# n = 5
# prize = np.array([50, 60, 40, 30, 20])
# weight = np.array([[1], [1], [1], [1], [1]])
# heuristics = heuristics_v2(prize, weight)
# print(heuristics)