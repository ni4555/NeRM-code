import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the average distance between all pairs of nodes
    avg_distance = np.mean(distance_matrix)

    # Hypothetical heuristic: Calculate the deviation from the average distance
    # Higher deviation means the edge is more unusual or "expensive" to traverse
    deviations = np.abs(distance_matrix - avg_distance)

    # Normalize the deviations to be between 0 and 1, where closer to 0 is more preferred
    normalized_deviations = (deviations - np.min(deviations)) / (np.max(deviations) - np.min(deviations))

    # Add a small constant to avoid division by zero and to make the heuristic smooth
    normalized_deviations += 1e-6

    # Invert the values to favor edges that are closer to the average (exploitation)
    heuristics = 1 / normalized_deviations

    # Adjust the heuristic to have a maximum value above the specified threshold
    max_value = np.max(heuristics)
    threshold = 10.604630532541204
    heuristics = np.clip(heuristics, None, threshold / max_value)

    return heuristics