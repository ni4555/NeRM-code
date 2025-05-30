import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check that the inputs are 2D tensors with matching dimensions
    if distance_matrix.ndim != 2 or demands.ndim != 1:
        raise ValueError("Invalid input shape. 'distance_matrix' should be 2D and 'demands' should be 1D.")
    if distance_matrix.shape[0] != distance_matrix.shape[1] or demands.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Mismatched dimensions. 'distance_matrix' should be n x n and 'demands' should be n.")
    
    # Initialize a tensor of zeros with the same shape as distance_matrix
    heuristics = torch.zeros_like(distance_matrix, dtype=torch.float32)
    
    # Normalize demands to sum to 1
    total_demand = demands.sum()
    normalized_demands = demands / total_demand
    
    # Define a small positive number for avoiding division by zero
    epsilon = 1e-10
    
    # Compute the heuristic values using a simple heuristic (e.g., inverse distance)
    heuristics = -1 / (distance_matrix + epsilon)
    
    # Adjust the heuristics by considering the demands
    # This could be a function of how much each customer's demand contributes to the desirability of an edge
    heuristics *= normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    return heuristics