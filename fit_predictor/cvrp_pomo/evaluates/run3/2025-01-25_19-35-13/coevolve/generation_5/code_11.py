import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics values
    # A simple heuristic could be the negative of the distance, as shorter distances are better
    # However, we can also take into account the normalized demand to prioritize heavier customers
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics