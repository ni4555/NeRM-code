import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the utility ratio for each item
    utility_ratio = prize / weight.sum(axis=1)
    
    # Calculate the adaptive stochastic sampling heuristic
    # Here we use a simple random sampling to demonstrate the concept,
    # in practice, a more complex adaptive strategy could be implemented
    random_sampling = np.random.rand(*prize.shape)
    
    # Combine utility ratio and random sampling to create a heuristic score
    heuristics = utility_ratio * random_sampling
    
    return heuristics