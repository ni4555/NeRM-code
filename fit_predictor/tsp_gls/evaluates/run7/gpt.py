import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    num_edges = distance_matrix.shape[0]
    precision_matrix = np.zeros_like(distance_matrix)
    
    # Define the precision heuristic matrix using the Manhattan distance
    heuristic_matrix = np.abs(distance_matrix - np.min(distance_matrix, axis=0) - 
                             np.min(distance_matrix, axis=1, keepdims=True))
    
    # Normalize the heuristic matrix to create a precision heuristic matrix
    heuristic_sum = np.sum(heuristic_matrix, axis=0, keepdims=True)
    precision_matrix = np.where(heuristic_sum == 0, 1, heuristic_matrix / heuristic_sum)
    
    # Introduce variability and perturbation to enhance exploration
    perturbation_factor = np.mean(precision_matrix) * 0.01
    perturbation = np.random.normal(0, perturbation_factor, precision_matrix.shape)
    precision_matrix += perturbation
    
    # Introduce a penalty for the longest edge in each row to discourage it from being included
    penalty = np.max(distance_matrix, axis=1, keepdims=True)
    precision_matrix = np.where(distance_matrix == penalty, np.inf, precision_matrix)
    
    # Ensure that the precision matrix remains positive
    precision_matrix = np.clip(precision_matrix, 0, np.inf)
    
    # Adjust the precision matrix based on the context of the problem
    num_cities = distance_matrix.shape[0]
    adjustment_factor = num_cities * (1 / (num_cities - 1))
    precision_matrix *= adjustment_factor
    
    return precision_matrix
