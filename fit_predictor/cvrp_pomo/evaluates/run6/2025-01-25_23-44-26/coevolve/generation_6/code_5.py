import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the demand penalty matrix
    demand_penalty_matrix = (1 - normalized_demands) * 1000  # Example penalty factor
    
    # Combine the inverse distance and demand penalty into the heuristic matrix
    heuristic_matrix = inv_distance_matrix - demand_penalty_matrix
    
    return heuristic_matrix