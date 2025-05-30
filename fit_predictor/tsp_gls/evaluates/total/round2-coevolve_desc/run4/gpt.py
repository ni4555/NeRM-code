import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    edge_usage = np.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(i+1, n):
            edge_usage[i, j] += 1
            edge_usage[j, i] += 1

    # Introduce randomness in usage data to avoid bias
    randomness_factor = np.random.rand(n, n) * 0.01
    edge_usage += randomness_factor

    # Non-linearly transform the adjusted distances
    adjusted_distances = np.sqrt(distance_matrix * (1 + 0.1 * (edge_usage / np.max(edge_usage))))

    # Introduce controlled perturbations with non-linear scaling
    perturbation = np.exp(np.random.normal(size=(n, n))) * 0.01
    adjusted_distances += perturbation

    # Ensure distances are within bounds to maintain consistency
    min_distance = 0.01
    max_distance = distance_matrix.max() * 0.9
    adjusted_distances = np.clip(adjusted_distances, min_distance, max_distance)

    # Introduce feedback mechanism based on edge properties
    symmetry_factor = np.ones_like(distance_matrix)
    for i in range(n):
        for j in range(n):
            if i != j:
                symmetry_factor[i, j] = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
    adjusted_distances *= symmetry_factor

    # Normalize the data to maintain invariance
    adjusted_distances = adjusted_distances / np.max(adjusted_distances) * distance_matrix.max()

    return adjusted_distances
