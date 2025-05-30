import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to have a range of [0, 1]
    distance_matrix_min = distance_matrix.min(dim=1, keepdim=True)[0]
    distance_matrix_max = distance_matrix.max(dim=1, keepdim=True)[0]
    normalized_distance_matrix = (distance_matrix - distance_matrix_min) / (distance_matrix_max - distance_matrix_min)
    
    # Normalize the demands to have a range of [0, 1]
    demands_min = demands.min()
    demands_max = demands.max()
    normalized_demands = (demands - demands_min) / (demands_max - demands_min)
    
    # Calculate the negative inverse of the normalized demands as part of the heuristic
    negative_inverse_demands = -1 / (normalized_demands + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Compute the heuristic values
    # The heuristic is designed to prefer edges with lower normalized distance and lower demands
    heuristic_values = (1 - normalized_distance_matrix) * negative_inverse_demands
    
    # Ensure the heuristic matrix is of the same shape as the input distance matrix
    assert heuristic_values.shape == distance_matrix.shape, "Heuristic matrix must have the same shape as the distance matrix."
    
    return heuristic_values