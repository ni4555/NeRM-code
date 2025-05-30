import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost of each edge as the product of the distance and the normalized demand
    potential_costs = distance_matrix * normalized_demands
    
    # Apply a simple heuristic: edges with lower potential cost are more promising
    # Here we subtract the potential costs from a large number to create negative values for undesirable edges
    heuristics = -potential_costs
    
    # To ensure that the matrix contains both positive and negative values, we add a small constant
    heuristics += 1e-6
    
    return heuristics