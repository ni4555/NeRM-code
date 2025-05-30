import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = len(prize)
    m = len(weight[0])
    
    # Calculate weighted ratios for each item
    weighted_ratios = prize / weight.sum(axis=1)
    
    # Use an adaptive dynamic sorting algorithm to sort items by their weighted ratios
    # For simplicity, we'll use a basic sorting algorithm here, but in a real scenario,
    # you might want to implement a more sophisticated sorting algorithm that adapts to the data
    sorted_indices = np.argsort(weighted_ratios)[::-1]
    
    # Implement an intelligent sampling mechanism to reduce the problem size
    # We'll take a random sample of items to consider, which can be improved with more sophisticated techniques
    num_samples = min(n, 10)  # You can adjust the number of samples as needed
    sampled_indices = np.random.choice(n, num_samples, replace=False)
    
    # Update the sorted indices to only include the sampled items
    sorted_indices = sorted_indices[sorted_indices.isin(sampled_indices)]
    
    # Use a greedy algorithm to determine the heuristics for each item
    # We'll assume that the greedy strategy is to take the top N items with the highest heuristics
    num_items_to_consider = min(n, 10)  # You can adjust the number of items to consider
    top_n_indices = sorted_indices[:num_items_to_consider]
    
    # The heuristics for each item is the ratio of the top N items
    top_n_weighted_ratios = weighted_ratios[top_n_indices]
    heuristics = np.array([1 if w in top_n_weighted_ratios else 0 for w in weighted_ratios])
    
    return heuristics

# Example usage:
# prize = np.array([60, 100, 120, 80, 70])
# weight = np.array([[1, 2], [1, 3], [2, 2], [1, 2], [2, 1]])
# print(heuristics_v2(prize, weight))