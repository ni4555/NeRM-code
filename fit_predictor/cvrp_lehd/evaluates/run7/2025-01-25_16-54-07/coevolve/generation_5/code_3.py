import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Calculate the heuristic for each edge
    # The heuristic is a function of distance and demand
    # We use a simple heuristic: distance * (1 + demand)
    heuristics = distance_matrix * (1 + normalized_demands)
    
    return heuristics