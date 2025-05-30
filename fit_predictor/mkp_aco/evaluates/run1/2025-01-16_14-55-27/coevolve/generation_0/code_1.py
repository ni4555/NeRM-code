import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure that the prize and weight arrays are both 2D with shape (n, m)
    if prize.ndim != 1 or weight.ndim != 2 or prize.shape[0] != weight.shape[0] or weight.shape[1] != 1:
        raise ValueError("Input arrays must have shape (n,) for prize and (n, 1) for weight.")
    
    # Calculate the utility score for each item by dividing the prize by the weight
    # In this case, since the weight is fixed to 1 for each item, the utility score is just the prize itself
    utility_scores = prize
    
    # Normalize the utility scores to make the heuristics relative to each other
    # This is a common step in heuristics to ensure that items are comparable
    max_utility = np.max(utility_scores)
    min_utility = np.min(utility_scores)
    # Avoid division by zero by setting the minimum utility to a small positive value
    min_utility = max(min_utility, 1e-10)
    normalized_scores = (utility_scores - min_utility) / (max_utility - min_utility)
    
    # Scale the normalized scores to the range [0, 1]
    heuristics = normalized_scores
    
    return heuristics