import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Calculate the cumulative sum of normalized demands for each row
    cumulative_demands = torch.cumsum(normalized_demands, dim=1)
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each edge, calculate the heuristic value
    # The heuristic is a function of the cumulative demand at the destination node
    # and the distance between the nodes. Here, we use a simple heuristic:
    # heuristics[i, j] = -distance_matrix[i, j] + cumulative_demands[j]
    heuristics = -distance_matrix + cumulative_demands
    
    return heuristics