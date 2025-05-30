import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    n = prize.shape[0]
    heuristics = np.zeros(n)
    
    # Calculate the total weight capacity
    total_weight_capacity = np.sum(weight, axis=1)
    
    # Initialize a matrix to store cumulative rewards
    cumulative_rewards = np.zeros((n, n))
    
    # Calculate cumulative rewards for each item
    for i in range(n):
        cumulative_rewards[i, i] = prize[i]
        for j in range(i + 1, n):
            cumulative_rewards[i, j] = prize[i] + prize[j]
    
    # Initialize a matrix to store cumulative weights
    cumulative_weights = np.zeros((n, n))
    
    # Calculate cumulative weights for each item
    for i in range(n):
        cumulative_weights[i, i] = weight[i, 0]
        for j in range(i + 1, n):
            cumulative_weights[i, j] = weight[i, 0] + weight[j, 0]
    
    # Initialize a matrix to store heuristic values
    heuristic_matrix = np.zeros((n, n))
    
    # Calculate heuristic values
    for i in range(n):
        for j in range(i + 1, n):
            heuristic_matrix[i, j] = cumulative_rewards[i, j] / cumulative_weights[i, j]
    
    # Apply adaptive dynamic heuristic adjustment
    for i in range(n):
        for j in range(i + 1, n):
            if heuristic_matrix[i, j] > heuristics[i]:
                heuristics[i] = heuristic_matrix[i, j]
            if heuristic_matrix[i, j] > heuristics[j]:
                heuristics[j] = heuristic_matrix[i, j]
    
    return heuristics
