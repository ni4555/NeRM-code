import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the demand per unit distance for each edge
    demand_per_unit_distance = demands.unsqueeze(1) / distance_matrix
    
    # Normalize by total capacity to give relative importance
    normalized_demand_per_unit_distance = demand_per_unit_distance / total_capacity
    
    # Create a penalty matrix for the edges based on demand
    penalty_matrix = torch.abs(normalized_demand_per_unit_distance)
    
    # Use a small positive constant to avoid division by zero
    epsilon = 1e-8
    distance_matrix = torch.clamp(distance_matrix, min=epsilon)
    
    # Calculate the heuristic for each edge as a function of distance and demand
    heuristic_matrix = penalty_matrix / distance_matrix
    
    return heuristic_matrix