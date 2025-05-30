import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize the demands to have a sum of 1
    normalized_demands = demands / vehicle_capacity
    
    # Create a demand matrix where each element is the product of the corresponding elements
    # from the distance matrix and the normalized demand vector.
    demand_matrix = distance_matrix * normalized_demands.unsqueeze(1)
    
    # The heuristic matrix will be negative for undesirable edges and positive for promising ones.
    # We can use the negative of the demand matrix as a heuristic, where lower values are better.
    # We add a small constant to avoid zeros for numerical stability.
    heuristic_matrix = -demand_matrix + 1e-8
    
    return heuristic_matrix