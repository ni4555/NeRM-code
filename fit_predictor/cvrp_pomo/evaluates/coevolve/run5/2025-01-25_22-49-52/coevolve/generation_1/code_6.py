import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics using the following formula:
    # heuristics[i][j] = distance[i][j] * (1 + normalized_demand[j])
    # This formula promotes edges with lower distance and higher demand
    heuristics = distance_matrix * (1 + normalized_demands.unsqueeze(1))
    
    return heuristics