import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized value for each item
    normalized_value = prize / np.sum(weight, axis=1, keepdims=True)
    
    # Incorporate stochastic sampling by adding a random perturbation to normalized value
    np.random.seed(0)  # Ensure reproducibility
    perturbation = np.random.normal(0, 0.1, normalized_value.shape)
    stochastic_normalized_value = normalized_value + perturbation
    
    # Rank items based on the adjusted normalized value
    rank = np.argsort(-stochastic_normalized_value, axis=0)
    
    # Calculate the probability of selection based on rank
    probability = 1 / (rank + 1)
    
    # Sum the probabilities across all dimensions for each item
    item_probabilities = np.sum(probability, axis=1)
    
    # Normalize the probabilities so that they sum to 1
    normalized_item_probabilities = item_probabilities / np.sum(item_probabilities)
    
    # Return the heuristics as an array where higher values indicate more promising items
    return normalized_item_probabilities