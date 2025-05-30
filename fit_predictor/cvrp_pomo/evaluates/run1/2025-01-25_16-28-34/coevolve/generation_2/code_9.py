import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize individual demands
    total_demand = demands.sum()
    
    # Normalize demands to fit the vehicle capacity (assumed to be 1.0 for simplicity)
    normalized_demands = demands / total_demand
    
    # Calculate the sum of distances for each row, which will be used to adjust the heuristic
    row_sum_distances = distance_matrix.sum(dim=1)
    
    # Calculate the heuristics based on normalized demand and adjusted distance
    # Here we assume a simple heuristic that promotes short paths with high demand
    heuristics = -normalized_demands * row_sum_distances
    
    # Add a small positive constant to ensure that no edge has a heuristic of zero
    # which might cause issues with certain optimization algorithms
    epsilon = 1e-10
    heuristics = heuristics + epsilon
    
    return heuristics