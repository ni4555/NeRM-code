import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demand vector does not include the depot's demand
    demands = demands[1:]
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands by the total capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the load for each edge as the product of distances and normalized demands
    load = (distance_matrix * normalized_demands.unsqueeze(1)).squeeze(1)
    
    # Define a constant for the heuristic to adjust the weight of demand
    demand_weight = 0.5
    
    # Calculate the heuristic value for each edge as a weighted sum of distance and load
    heuristic_values = -demand_weight * distance_matrix + (1 - demand_weight) * load
    
    return heuristic_values