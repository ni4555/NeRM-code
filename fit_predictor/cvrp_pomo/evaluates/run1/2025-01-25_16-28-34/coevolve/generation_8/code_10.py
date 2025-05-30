import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_capacity
    
    # Compute the initial heuristic values based on demand
    heuristics[distance_matrix != 0] = -normalized_demands[distance_matrix != 0]
    
    # Vectorized implementation of a simple heuristic: closer nodes are more promising
    # We subtract the distance because shorter distances are more promising
    heuristics[distance_matrix != 0] -= distance_matrix[distance_matrix != 0]
    
    # Adjust the heuristic values to ensure they are non-negative
    heuristics[distance_matrix != 0] = torch.clamp(heuristics[distance_matrix != 0], min=0)
    
    return heuristics