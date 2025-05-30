import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of demands for each edge
    edge_demands = distance_matrix * normalized_demands
    
    # Compute a heuristic based on the edge demands
    # For simplicity, we'll use the sum of demands as the heuristic
    # Negative values for undesirable edges and positive for promising ones
    heuristics = edge_demands.sum(dim=1)
    
    return heuristics