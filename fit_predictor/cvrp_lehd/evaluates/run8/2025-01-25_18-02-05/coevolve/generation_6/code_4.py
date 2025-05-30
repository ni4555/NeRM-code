import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized inverse of the demands
    normalized_demands = 1 / (demands / demands.sum())
    
    # Compute the heuristic value for each edge
    heuristics = distance_matrix * normalized_demands
    
    return heuristics