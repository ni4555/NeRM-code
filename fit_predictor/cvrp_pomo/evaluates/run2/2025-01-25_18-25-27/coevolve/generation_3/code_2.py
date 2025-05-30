import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = torch.sum(demands)
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Compute the potential cost of each edge based on distance and demand
    # Negative values indicate undesirable edges (high cost), positive values indicate promising edges (low cost)
    edge_potential = distance_matrix * normalized_demands
    
    return edge_potential