import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Normalize the prize and weight by their max values in each dimension
    prize_normalized = prize / prize.max()
    weight_normalized = weight / weight.max()
    
    # Calculate the dynamic weighted ratio index
    dynamic_weighted_ratio = prize_normalized / weight_normalized
    
    # Apply adaptive probabilistic sampling to the normalized prizes
    adaptive_prob_sampling = np.random.rand(len(prize))
    
    # Calculate the heuristic value for each item based on the dynamic weighted ratio and probability
    heuristics = dynamic_weighted_ratio * adaptive_prob_sampling
    
    return heuristics