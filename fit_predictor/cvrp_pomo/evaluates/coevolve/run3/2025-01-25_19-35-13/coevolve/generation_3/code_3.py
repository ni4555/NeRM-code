import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the potential function based on distance and demand
    potential = -normalized_distance_matrix * demand_normalized
    
    # Incorporate epsilon to prevent division by zero
    epsilon = 1e-8
    potential = torch.clamp(potential, min=epsilon)
    
    # Calculate the heuristics matrix
    heuristics_matrix = potential.sum(dim=1)
    
    return heuristics_matrix