import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    # The heuristic is a combination of the normalized demand and distance
    # Negative values for undesirable edges, positive values for promising ones
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics