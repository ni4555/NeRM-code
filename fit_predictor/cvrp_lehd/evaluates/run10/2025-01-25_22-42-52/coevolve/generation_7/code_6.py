import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands to match the range of the distance matrix
    normalized_demands = demands / total_capacity
    
    # Create a matrix of customer demands
    demand_matrix = torch.full_like(distance_matrix, fill_value=normalized_demands)
    
    # Calculate the difference between the distance matrix and the demand matrix
    # This will give us a measure of the "promise" of each edge
    promise_matrix = distance_matrix - demand_matrix
    
    # We want to return positive values for promising edges and negative values for undesirable ones
    # We can do this by adding a small constant to the negative values to ensure they are negative
    small_constant = 1e-6
    promise_matrix = torch.where(promise_matrix < 0, promise_matrix - small_constant, promise_matrix)
    
    return promise_matrix