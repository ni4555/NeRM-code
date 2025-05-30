import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Calculate the total capacity for the knapsack
    total_capacity = np.sum(weight, axis=1)
    # Initialize heuristic values
    heuristics = np.zeros(n)
    # Loop over each item
    for i in range(n):
        # Calculate the reward to weight ratio for the current item
        reward_to_weight_ratio = prize[i] / total_capacity[i]
        # Calculate the contribution of the item to the overall diversity
        diversity_contribution = np.abs(np.sum(weight[:, :i]) - np.sum(weight[:, i+1:]))
        # Update the heuristic value based on the reward-to-weight ratio and diversity
        heuristics[i] = reward_to_weight_ratio + diversity_contribution
    # Normalize the heuristic values to ensure they sum to the total capacity
    heuristics /= np.sum(heuristics)
    return heuristics
