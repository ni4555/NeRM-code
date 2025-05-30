import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand vector
    normalized_demands = demands / demands.sum()
    
    # Calculate the demand contribution for each edge
    edge_demands = torch.outer(normalized_demands, normalized_demands)
    
    # Calculate the heuristic values by subtracting the distance (to encourage short routes)
    heuristics = -distance_matrix + torch.sum(edge_demands, dim=0)
    
    # Ensure that the heuristics matrix is of the same shape as the distance matrix
    assert heuristics.shape == distance_matrix.shape
    
    return heuristics