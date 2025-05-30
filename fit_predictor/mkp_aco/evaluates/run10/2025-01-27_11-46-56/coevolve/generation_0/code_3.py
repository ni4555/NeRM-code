import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    if m == 1:
        return prize / weight[:, 0]
    else:
        # Calculate the "density" of each item
        density = prize / weight.sum(axis=1)
        # Perform a stochastic solution sampling based on density
        random_samples = np.random.rand(n)
        return (random_samples > density).astype(int)
