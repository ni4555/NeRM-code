import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the demand for each edge (i, j)
    edge_demands = torch.abs(distance_matrix) * demands
    
    # Normalize the demand for each edge by the total demand
    normalized_demands = edge_demands / total_demand
    
    # Create a penalty for edges that exceed the vehicle capacity
    capacity_penalty = (normalized_demands > 1.0).to(torch.float32)
    
    # Subtract the penalty from the normalized demand to make it negative for undesirable edges
    heuristics = normalized_demands - capacity_penalty
    
    return heuristics