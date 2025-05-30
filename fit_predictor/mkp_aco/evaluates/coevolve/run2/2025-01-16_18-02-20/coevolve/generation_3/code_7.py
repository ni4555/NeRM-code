import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Prioritize items based on their value-to-weight ratio
    heuristics = value_to_weight_ratio
    
    # Apply adaptive stochastic sampling to the heuristics
    # Here, we assume a simple adaptive mechanism: increase the heuristic if the ratio is above a threshold
    threshold = np.percentile(value_to_weight_ratio, 75)  # Example threshold based on the 75th percentile
    heuristics[heuristics > threshold] *= 1.5  # Increase the heuristic by 50% for high ratios
    
    # Dynamic weight constraint adaptation is not explicitly implemented as it requires knowledge of the knapsack capacity
    # and the current state of the knapsack, which is not provided. However, this could be added if the necessary information is available.
    
    # Iterate over items and select the best one to maximize the prize while adhering to the weight constraints
    # Since we do not have the full resolution of the problem (i.e., the knapsack capacity and the weight constraints of each knapsack),
    # we will simply return the heuristics as is, which are based on the value-to-weight ratio and the adaptive stochastic sampling.
    
    return heuristics