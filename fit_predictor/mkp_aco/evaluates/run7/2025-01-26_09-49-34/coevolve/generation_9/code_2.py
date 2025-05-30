import numpy as np
import numpy as np
from scipy.stats import multivariate_normal

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = weight.shape
    num_samples = 1000
    item_means = np.mean(weight, axis=0)
    item_covariances = np.cov(weight, rowvar=False)
    max_prize = np.max(prize)
    heuristics = np.zeros(n)
    
    for i in range(n):
        # Generate random samples around the mean
        samples = multivariate_normal.rvs(item_means, item_covariances, num_samples)
        # Calculate fitness for each sample
        fitness = (prize[i] * np.exp(-np.sum(samples, axis=1) / max_prize))
        # Normalize fitness for the item
        heuristics[i] = np.sum(fitness) / num_samples
    
    return heuristics
