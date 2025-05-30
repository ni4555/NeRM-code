import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge
    # This is a simple heuristic that considers the demand of the destination node
    edge_potential = distance_matrix * normalized_demands
    
    # Subtract the demand from the potential to create a heuristic value
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristics = edge_potential - normalized_demands
    
    return heuristics