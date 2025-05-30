import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands to use in the heuristic
    cumsum_demands = torch.cumsum(demands, dim=0)
    
    # Calculate the heuristic as the negative of the sum of distances and the cumulative demand
    # The heuristic for an edge from node i to node j is: -sum of distances from i to j and cumulative demand up to j
    heuristics = -torch.sum(distance_matrix, dim=1) + cumsum_demands
    
    return heuristics