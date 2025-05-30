import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector to the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Compute the total demand of each edge (i, j)
    edge_demands = (distance_matrix * demands.unsqueeze(1)).sum(1)
    
    # Calculate the heuristic value for each edge as the normalized demand of the edge
    heuristics = normalized_demands * edge_demands
    
    # Return the heuristics matrix
    return heuristics