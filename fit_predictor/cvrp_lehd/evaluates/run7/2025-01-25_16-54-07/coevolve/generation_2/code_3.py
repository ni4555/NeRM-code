import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum().item()
    
    # Normalize the demand vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the sum of products of normalized demands and distance matrix
    heuristics = normalized_demands.view(-1, 1) * distance_matrix
    
    # Add negative values to make undesirable edges less promising
    undesirable_edges = distance_matrix < 0
    heuristics[undesirable_edges] = -1 * torch.abs(heuristics[undesirable_edges])
    
    return heuristics