import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This heuristic assumes that the distance matrix is symmetric and the diagonal is full of zeros.
    # The "badness" score for an edge is inversely proportional to its distance.
    # We use the minimum distance in the matrix to normalize the scores so that the smallest distance
    # gets the maximum score (e.g., a score of 1.0), and larger distances get lower scores.
    min_distance = np.min(distance_matrix[distance_matrix > 0])
    # Invert the distance to get a "badness" score (where smaller distances are better)
    badness_scores = 1.0 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    # Normalize the scores by dividing by the min_distance
    normalized_scores = badness_scores / min_distance
    return normalized_scores