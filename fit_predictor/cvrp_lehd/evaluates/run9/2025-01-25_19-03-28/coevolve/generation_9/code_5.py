import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the size of the distance matrix
    n = distance_matrix.size(0)
    
    # Create a demand matrix where the demand of each customer is normalized
    demand_matrix = demands / demands.sum()
    
    # Create a matrix for the negative weighted distances
    negative_weighted_distance_matrix = -distance_matrix
    
    # Calculate the heuristics matrix by multiplying the demand matrix with the negative weighted distance matrix
    heuristics_matrix = negative_weighted_distance_matrix * demand_matrix
    
    # Sum along the rows to get the total heuristics for each edge
    heuristics_matrix = heuristics_matrix.sum(dim=1, keepdim=True)
    
    return heuristics_matrix