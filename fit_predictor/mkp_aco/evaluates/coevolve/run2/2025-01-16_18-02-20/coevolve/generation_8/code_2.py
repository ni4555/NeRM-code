import numpy as np
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max to avoid overflow
    return exp_x / np.sum(exp_x, axis=0)

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize by dividing by the weight (which is 1 for each dimension)
    normalized_ratio = prize / weight
    
    # Apply softmax to get the probabilities
    probabilities = softmax(normalized_ratio)
    
    # Return the probabilities as the heuristic values
    return probabilities