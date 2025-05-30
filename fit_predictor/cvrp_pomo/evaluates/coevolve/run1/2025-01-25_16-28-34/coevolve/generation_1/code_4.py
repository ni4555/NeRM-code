import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the maximum distance to avoid large values
    distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Calculate the cumulative demand for each edge
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Compute the heuristics based on the normalized distance and cumulative demand
    heuristics = -distance_matrix + cumulative_demand
    
    return heuristics