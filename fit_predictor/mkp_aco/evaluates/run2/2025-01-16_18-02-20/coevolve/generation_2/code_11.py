import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize based on the total prize
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the normalized weight for each item
    normalized_weight = weight / np.sum(weight, axis=1)[:, np.newaxis]
    
    # Compute the normalized value for each item
    normalized_value = normalized_prize * normalized_weight
    
    # Rank the items based on normalized value
    rank = np.argsort(normalized_value, axis=1)[:, ::-1]
    
    # Initialize heuristic scores
    heuristics = np.zeros_like(prize, dtype=float)
    
    # Iterate over each item and calculate its heuristic score
    for i in range(rank.shape[0]):
        # Calculate the number of items that can be included while satisfying the weight constraint
        current_weight = np.sum(weight[rank[i]] < 1, axis=1)
        max_included = np.sum(current_weight)
        
        # Update the heuristic score based on the potential to include the item
        heuristics[i] = max_included / len(rank[i])
    
    return heuristics