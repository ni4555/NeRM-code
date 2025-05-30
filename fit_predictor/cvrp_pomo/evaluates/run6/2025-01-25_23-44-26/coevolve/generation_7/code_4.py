import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to create an inverse distance heuristic
    inv_distance = 1.0 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a demand-penalty matrix to deter overloading vehicles
    demand_penalty = normalized_demands * 1000  # Adjust the penalty factor as needed
    
    # Combine the inverse distance and demand-penalty to get the heuristic matrix
    heuristic_matrix = inv_distance - demand_penalty
    
    return heuristic_matrix