import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix to have non-negative values
    distance_matrix = distance_matrix - distance_matrix.min(dim=0, keepdim=True)[0]
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the demand per unit distance for each edge
    demand_per_unit_distance = demands / distance_matrix
    
    # Calculate the heuristics based on demand per unit distance
    heuristics = -demand_per_unit_distance
    
    # Normalize the heuristics to sum to the total vehicle capacity
    heuristics /= heuristics.sum()
    
    # Ensure that the sum of heuristics is close to the total capacity
    heuristics += (total_capacity - heuristics.sum())
    
    return heuristics