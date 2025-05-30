import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values as the difference between the negative of the distance and the normalized demand
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1)
    
    return heuristics