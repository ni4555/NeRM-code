import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the value-to-weight ratio for each item
    normalized_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to sum to 1
    normalized_ratio /= normalized_ratio.sum()
    
    # Initialize a list to store the heuristics for each item
    heuristics = np.zeros_like(prize)
    
    # Initialize a random number generator
    rng = np.random.default_rng()
    
    # Iterate over each item and assign a heuristic based on the normalized ratio
    for i in range(prize.shape[0]):
        # Generate a random number to simulate probabilistic item selection
        random_number = rng.random()
        
        # Calculate the cumulative probability up to the current item
        cumulative_probability = np.cumsum(normalized_ratio)
        
        # Assign the heuristic based on whether the random number falls within the cumulative probability
        heuristics[i] = 1 if random_number < cumulative_probability[i] else 0
    
    return heuristics