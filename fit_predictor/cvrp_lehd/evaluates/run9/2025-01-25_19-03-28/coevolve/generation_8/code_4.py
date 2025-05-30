import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand difference for each edge (i, j)
    demand_diff = demands - demands.unsqueeze(0)
    
    # Calculate the negative demand difference for edges with higher demands
    negative_demand_diff = -torch.abs(demand_diff)
    
    # Calculate the distance-based penalty for each edge
    distance_penalty = distance_matrix
    
    # Combine the demand difference and distance penalty
    combined_penalty = negative_demand_diff + distance_penalty
    
    # Normalize the combined penalty by the maximum value
    normalized_penalty = combined_penalty / combined_penalty.max()
    
    # Convert to a tensor of the same shape as the distance matrix
    heuristics = normalized_penalty.unsqueeze(0)
    
    return heuristics