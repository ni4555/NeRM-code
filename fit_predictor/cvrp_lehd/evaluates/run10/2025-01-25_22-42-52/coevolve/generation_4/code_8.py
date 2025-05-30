import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance_matrix and demands are of the correct shape and type
    if not (distance_matrix.ndim == 2 and demands.ndim == 1):
        raise ValueError("Invalid input shapes.")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if demands.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Demands vector length must match the number of nodes in the distance matrix.")
    
    # Calculate the total vehicle capacity (assuming it's a single value for all vehicles)
    vehicle_capacity = demands.sum()
    
    # Normalize the demands by the vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Create a mask for the diagonal elements of the distance matrix
    diagonal_mask = torch.eye(distance_matrix.shape[0], dtype=torch.bool)
    
    # Create a mask for edges with zero distance (self-loops)
    zero_distance_mask = (distance_matrix == 0) & ~diagonal_mask
    
    # Initialize the heuristic matrix with negative infinity for self-loops and zero distances
    heuristic_matrix = -torch.inf * zero_distance_mask.type(torch.float32)
    heuristic_matrix[~zero_distance_mask] = 0
    
    # Calculate the heuristics for the remaining edges
    # The heuristic could be a function of distance and demand
    # Here, we use a simple heuristic: the product of distance and normalized demand
    heuristic_matrix[~zero_distance_mask] = distance_matrix[~zero_distance_mask] * normalized_demands
    
    return heuristic_matrix