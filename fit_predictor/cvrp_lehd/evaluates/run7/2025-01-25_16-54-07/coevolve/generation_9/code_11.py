import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each edge
    cumulative_demand = (demands.unsqueeze(0) * demands.unsqueeze(1)).sum(dim=2)
    
    # Normalize the cumulative demand by the total vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()
    
    # Calculate the heuristic value for each edge
    heuristic_values = distance_matrix - normalized_demand
    
    return heuristic_values