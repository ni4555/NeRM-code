import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand
    normalized_demand = demands / demands.sum()

    # Calculate the heuristic for each edge
    heuristics = distance_matrix * normalized_demand
    
    # Adjust heuristics for overloading: reduce heuristics for edges leading to overloading
    total_demand = torch.cumsum(normalized_demand, dim=0)
    overloading_threshold = demands.new_zeros(1).fill_(1)
    heuristics[total_demand > 1] *= -1  # Mark overloading edges with negative heuristics
    
    return heuristics