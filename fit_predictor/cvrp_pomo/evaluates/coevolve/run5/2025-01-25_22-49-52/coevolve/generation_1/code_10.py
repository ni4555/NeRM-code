import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the potential heuristics for each edge
    # Here we use a simple heuristic that considers the demand and distance
    # We subtract the normalized demand from the distance to get a negative heuristic
    # for promising edges (shorter distances with higher demand)
    heuristics = distance_matrix - normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics