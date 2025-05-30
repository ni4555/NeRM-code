import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the distance matrix by the maximum distance to avoid large negative values
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Calculate the negative of the demands to promote their selection
    negative_demands = -demands
    
    # Calculate the load balance factor for each customer
    load_balance_factor = torch.abs(negative_demands / torch.sum(negative_demands))
    
    # Combine the factors to create the heuristic
    heuristic_matrix = normalized_distance_matrix + load_balance_factor
    
    # Ensure that the heuristic matrix is not too large to avoid dominated solutions
    # by clamping the values to a maximum of 1
    heuristic_matrix = torch.clamp(heuristic_matrix, max=1)
    
    return heuristic_matrix