import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming it's the sum of all demands)
    total_capacity = demands.sum()
    
    # Normalize demands to the range [0, 1] based on the total capacity
    normalized_demands = demands / total_capacity
    
    # Compute the heuristics based on normalized demands
    # The heuristic for each edge is the negative of the normalized demand of the destination node
    heuristics = -normalized_demands[distance_matrix != 0]
    
    # Expand the heuristics tensor to match the shape of the distance matrix
    heuristics = heuristics.view(distance_matrix.shape)
    
    # Fill the diagonal with zeros, as we don't want to consider the depot to itself
    torch.fill_diagonal_(heuristics, 0)
    
    return heuristics