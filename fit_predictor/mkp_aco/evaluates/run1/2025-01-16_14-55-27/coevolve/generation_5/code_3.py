import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight of each item
    total_weight = weight.sum(axis=1)
    
    # Calculate the total prize for each item
    total_prize = prize.sum(axis=1)
    
    # Advanced ratio analysis: prize to weight ratio
    ratio = total_prize / total_weight
    
    # Multi-criteria ranking system
    # Here we're considering only the prize to weight ratio for simplicity,
    # but in a real-world scenario, this could be extended to include other criteria.
    # Sort the ratios in descending order to prioritize higher ratios
    sorted_indices = np.argsort(ratio)[::-1]
    
    # Generate the heuristics array with the highest ratio as the most promising
    heuristics = np.zeros_like(prize)
    heuristics[sorted_indices] = 1
    
    return heuristics