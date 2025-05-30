import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the potential of each edge as a product of the inverse of distance and normalized demand
    potential = torch.reciprocal(distance_matrix) * normalized_demands
    
    # Ensure that the diagonal is zero (no self-loops)
    torch.fill_diagonal(potential, 0)
    
    # Apply a small epsilon to avoid division by zero (for edges with distance 0)
    epsilon = 1e-8
    potential = torch.clamp(potential, min=epsilon)
    
    return potential