import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the utility ratio for each item
    utility_ratio = prize / weight.sum(axis=1)
    
    # Calculate the ratio of the prize to the weight for each dimension
    prize_to_weight_ratio = prize / weight
    
    # Define a multi-criteria ranking system
    multi_criteria_score = (utility_ratio + prize_to_weight_ratio.mean(axis=1)) / 2
    
    # Rank items based on the multi-criteria score
    ranked_indices = np.argsort(-multi_criteria_score)
    
    # Create an array that represents the heuristics for each item
    heuristics = np.zeros(prize.shape)
    heuristics[ranked_indices] = 1
    
    return heuristics