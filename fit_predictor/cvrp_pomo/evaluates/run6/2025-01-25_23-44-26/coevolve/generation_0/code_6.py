import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Calculate the negative of the distance matrix for the heuristic
    negative_distance_matrix = -distance_matrix
    
    # Calculate the sum of the products of distances and normalized demands
    # This is equivalent to the heuristic value for each edge
    heuristic_values = torch.matmul(negative_distance_matrix, normalized_demands)
    
    return heuristic_values