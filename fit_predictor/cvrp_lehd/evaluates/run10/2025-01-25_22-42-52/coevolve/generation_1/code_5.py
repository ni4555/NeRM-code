import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the negative of the distance matrix to create a heuristic
    # where shorter distances are more promising
    negative_distance = -distance_matrix
    
    # Calculate the total demand to normalize by
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity to create a demand-based heuristic
    normalized_demands = demands / total_demand
    
    # Combine the distance-based heuristic with the demand-based heuristic
    combined_heuristic = negative_distance * normalized_demands
    
    # Ensure that the diagonal elements are zero to avoid considering the depot as a customer
    torch.fill_diagonal_(combined_heuristic, 0)
    
    return combined_heuristic