import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Step 1: Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Step 2: Implement a dynamic sorting algorithm to adapt to the weighted ratios
    # Here, we'll use a simple version of the dynamic sorting algorithm based on the weighted ratio
    item_promise = np.argsort(weighted_ratio)[::-1]  # Sort items by their weighted ratio in descending order
    
    # Step 3: Employ an intelligent sampling mechanism
    # Here, we will sample a subset of items that have the highest weighted ratio
    num_items_to_sample = int(n * 0.1)  # For example, we sample 10% of the items
    sampled_items = item_promise[:num_items_to_sample]
    
    # Step 4: Combine greedy algorithm with heuristic-based search strategies
    # Create an initial heuristic array based on the sorted weighted ratio
    heuristics = np.zeros(n)
    for i in sampled_items:
        heuristics[i] = 1  # Assume all sampled items are promising
    
    # Apply additional heuristic-based search strategies
    # For instance, we could use a heuristic that looks at the total weight and prize after including items
    # This could be done by trying to include the next most promising item without exceeding the weight limits
    # Here, we use a simple greedy approach to include as many items as possible
    for i in item_promise:
        if np.all(weight[i] <= 1) and heuristics[i] == 0:  # Check if item can be added without exceeding the constraints
            # Assume adding this item is promising, as we're using a greedy approach
            heuristics[i] = 1
            # Update the remaining weight to reflect the inclusion of this item
            weight[:, weight[i] > 0] -= weight[i]
            weight[i] = 0  # Set the weight to zero after including it
    
    return heuristics

# Example usage:
prize_example = np.array([60, 100, 120, 80])
weight_example = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
heuristic_example = heuristics_v2(prize_example, weight_example)
print(heuristic_example)