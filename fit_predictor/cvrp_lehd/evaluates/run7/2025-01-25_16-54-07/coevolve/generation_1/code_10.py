import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand normalized by the vehicle capacity
    total_demand = demands.sum()
    
    # Calculate the heuristics for each edge based on the formula:
    # heuristics[i, j] = distance[i, j] - demands[i] * (demands[j] / total_demand)
    heuristics = distance_matrix - (demands[:, None] * (demands[None, :] / total_demand))
    
    # Convert negative values to zero to mark undesirable edges
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics