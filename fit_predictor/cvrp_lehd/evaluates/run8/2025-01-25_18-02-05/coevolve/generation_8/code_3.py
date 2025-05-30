import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the heuristic values using a simple method that considers distance and demand
    # Negative heuristic values for edges to the depot (index 0) and for high demand
    negative_heuristics = -distance_matrix + (demands > demands.mean()).float() * 10
    
    # Calculate the heuristic values for other edges
    positive_heuristics = distance_matrix + (demands < demands.mean()).float() * 10
    
    # Combine negative and positive heuristics into a single tensor
    heuristics = torch.where(distance_matrix == 0, negative_heuristics, positive_heuristics)
    
    return heuristics