import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize heuristic values to be the prize value, as the initial best estimate
    heuristics = np.copy(prize)
    
    # Compute a density function for the weights, favoring lower density areas
    density = 1.0 / (weight.sum(axis=1, keepdims=True) ** 2)
    
    # Adaptive heuristic based on a balance of prize value and weight density
    for i in range(len(heuristics)):
        for j in range(len(heuristics)):
            if j != i:
                heuristics[j] *= density[i]
                heuristics[j] += prize[i]
                
    # Incorporate anti-local optima mechanism
    # By slightly adjusting the density for anti-local optima, encouraging new paths
    for i in range(len(heuristics)):
        if weight[i, 0] < weight.sum(axis=0) * 0.01:  # Anti-local optima mechanism trigger
            heuristics *= (1.0 + 0.1 * (np.random.rand(len(heuristics)) - 0.5))
            break

    # Integrate adaptive stochastic sampling by slightly altering heuristic values
    # Introducing random perturbation around the calculated heuristics
    randomperturbation = 0.01 * (np.random.rand(len(heuristics)) - 0.5)
    heuristics += randomperturbation

    return heuristics
