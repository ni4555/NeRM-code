import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Add a small value to avoid division by zero
    
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Apply the demand-penalty mechanism
    demand_penalty = demands * 0.1  # Adjust the penalty factor as needed
    
    # Calculate the initial heuristic value based on inverse distance and demand-penalty
    heuristic_values = inv_distance_matrix * (1 - normalized_demands) + demand_penalty
    
    return heuristic_values