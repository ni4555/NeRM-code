import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = len(prize)
    m = len(weight[0])
    
    # Step 1: Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / np.sum(weight, axis=1)
    
    # Step 2: Dynamic multi-criteria sorting
    # For simplicity, we are only considering value-to-weight ratio, but in a full implementation,
    # you could add additional criteria and use a more complex sorting mechanism.
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Step 3: Normalize the heuristics
    # Normalize based on the maximum ratio to ensure the heuristics are on the same scale
    max_ratio = np.max(value_to_weight_ratio)
    normalized_heuristics = value_to_weight_ratio / max_ratio
    
    # Step 4: Adaptive stochastic sampling
    # This step is conceptual and depends on the specific algorithm used. Here, we'll just repeat
    # the normalization step as a placeholder for the adaptive sampling strategy.
    adaptive_stochastic_sampling = np.random.choice(n, size=int(n / 2), replace=False)
    for i in adaptive_stochastic_sampling:
        normalized_heuristics[i] = np.random.random()
    
    # The final heuristics array is the normalized heuristics, which is already of shape (n,)
    return normalized_heuristics

# Example usage:
# prize = np.array([10, 30, 20])
# weight = np.array([[1, 1], [2, 1], [1, 2]])
# print(heuristics_v2(prize, weight))