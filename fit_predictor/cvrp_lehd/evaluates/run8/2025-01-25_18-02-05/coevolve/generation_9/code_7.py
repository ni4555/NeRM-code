import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    # The heuristic is a combination of the normalized demand and the distance
    # In this case, we use a simple heuristic: -distance + demand
    # Negative distance is used to favor shorter paths
    # Demand is used to favor routes with higher demand
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    return heuristics