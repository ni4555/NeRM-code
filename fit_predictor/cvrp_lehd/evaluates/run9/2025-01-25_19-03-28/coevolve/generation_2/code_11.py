import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to have a range between 0 and 1
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the sum of demands for each node
    sum_of_demands = demands.sum(dim=0)
    
    # Calculate the heuristics using the sum of demands and the normalized distance matrix
    heuristics = -sum_of_demands * distance_matrix
    
    return heuristics