import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize weight data
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate the heuristic values
    heuristic_values = normalized_weight * prize
    
    # Implement adaptive sampling
    num_items = prize.shape[0]
    sample_size = min(num_items, 100)  # Example of adaptive sampling size
    sampled_indices = np.random.choice(num_items, sample_size, replace=False)
    sampled_heuristics = heuristic_values[sampled_indices]
    adaptive_heuristic = np.mean(sampled_heuristics)
    
    # Implement dynamic fitness assessment
    dynamic_fitness = heuristic_values / (1 + np.abs(heuristic_values - adaptive_heuristic))
    
    # Combine with local search technique (e.g., local greedy selection)
    selected_indices = np.argsort(dynamic_fitness)[::-1]
    heuristics = np.zeros(num_items)
    total_weight = np.zeros(m)
    for i in selected_indices:
        if np.all(total_weight < 1):
            heuristics[i] = 1
            total_weight += weight[i]
    
    return heuristics
