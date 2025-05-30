import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demands vector for normalization
    total_demand = demands.sum()
    
    # Normalize the demands to the range [0, 1]
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics as a function of the normalized demands
    # The heuristic is a combination of the normalized demand and the distance
    # Negative values are assigned to edges to indicate undesirable edges
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * demands.unsqueeze(0)
    
    return heuristics