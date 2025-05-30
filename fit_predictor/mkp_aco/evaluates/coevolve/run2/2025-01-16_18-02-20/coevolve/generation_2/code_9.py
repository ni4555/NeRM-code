import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Normalize the prize by the sum of the weights for each item
    normalized_value = prize / weight.sum(axis=1)
    
    # Incorporate stochastic sampling to adjust the normalized values
    # For simplicity, let's use a random perturbation
    np.random.seed(0)  # Setting a seed for reproducibility
    random_perturbation = np.random.normal(0, 0.1, normalized_value.shape)
    adjusted_normalized_value = normalized_value + random_perturbation
    
    # Rank the adjusted normalized values
    rank = np.argsort(-adjusted_normalized_value)  # Descending order
    
    # Generate a heuristic score for each item
    heuristic = np.zeros(n)
    heuristic[rank] = np.arange(1, n + 1)  # Higher rank, higher score
    
    return heuristic