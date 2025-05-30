import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each node
    demand_sums = normalized_demands.cumsum(dim=0)
    
    # Calculate the heuristic values
    # Positive values for promising edges and negative values for undesirable edges
    heuristics = distance_matrix + (demand_sums * distance_matrix)
    
    # Enforce vehicle capacity constraint by penalizing heavily loaded edges
    capacity_penalty = (demands > 1).float() * 1000  # Arbitrary large number for heavily loaded edges
    heuristics = heuristics + capacity_penalty
    
    return heuristics