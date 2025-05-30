import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands tensor is broadcastable to the shape of distance_matrix
    demands = demands.view(-1, 1)
    
    # Compute the demand difference
    demand_diff = demands - demands.t()
    
    # Add a small epsilon to avoid division by zero in case of equal demands
    epsilon = 1e-8
    demand_diff = torch.clamp(demand_diff, min=epsilon)
    
    # Compute the heuristic values
    heuristic_values = -distance_matrix + (1 - 2 * epsilon) * demand_diff
    
    return heuristic_values