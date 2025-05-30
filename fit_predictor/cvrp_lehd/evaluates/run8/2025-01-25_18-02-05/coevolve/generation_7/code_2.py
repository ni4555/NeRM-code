import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics as the negative of the distance (undesirable edges)
    # and add a small positive value for promising edges
    heuristics = -distance_matrix + 0.1 * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    return heuristics