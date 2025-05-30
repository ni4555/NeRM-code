import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the lower the heuristic value, the better the edge.
    # For the sake of this example, let's implement a simple heuristic.
    # We will use the distance from the first city (index 0) to all other cities.
    # In a more complex scenario, you would replace this with more sophisticated logic.
    
    # Get the distance from the first city to all others
    min_distances = np.min(distance_matrix, axis=0)
    
    # Return the negated distance as the heuristic value (lower is better)
    # Note that we use the negation since most evolutionary algorithms prefer
    # to minimize the heuristic values
    return -min_distances