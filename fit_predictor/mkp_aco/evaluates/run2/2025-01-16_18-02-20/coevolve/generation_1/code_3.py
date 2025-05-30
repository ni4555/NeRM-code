import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic for demonstration purposes: the higher the prize, the more promising the item
    # This is a placeholder for a more complex heuristic that could involve adaptive stochastic sampling and metaheuristics
    heuristics = prize / np.sum(weight, axis=1)
    return heuristics