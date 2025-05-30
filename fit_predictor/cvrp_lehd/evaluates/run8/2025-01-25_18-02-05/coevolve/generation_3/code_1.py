import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum().item()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    # We use the following heuristic: demand of the destination node minus the distance to the next node
    heuristics = normalized_demands - distance_matrix

    return heuristics