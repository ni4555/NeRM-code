import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the ratio of prize to weight for each item
    prize_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Use cumulative performance metrics to refine the ranking
    cumulative_prize = np.cumsum(prize)
    cumulative_weight = np.cumsum(weight.sum(axis=1))
    cumulative_ratio = cumulative_prize / cumulative_weight
    
    # Combine the ratio analysis with the multi-criteria ranking system
    combined_ranking = cumulative_ratio * prize_to_weight_ratio
    
    # Return the ranking for each item
    return combined_ranking