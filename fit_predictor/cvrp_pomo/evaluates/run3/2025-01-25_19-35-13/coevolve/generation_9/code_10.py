import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Calculate the total distance of the matrix
    total_distance = (distance_matrix ** 2).sum() / 2
    
    # Normalize demands to the total vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the heuristic values based on normalized demands
    # The heuristic can be any function that estimates the desirability of an edge
    # Here, we use a simple heuristic: the negative of the distance multiplied by the demand
    heuristics = -distance_matrix * normalized_demands
    
    # Optionally, you can add more complex heuristics here
    
    return heuristics