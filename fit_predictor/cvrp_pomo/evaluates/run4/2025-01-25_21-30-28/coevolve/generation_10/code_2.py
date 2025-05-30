import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inverse_distance = 1.0 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    
    # Calculate the demand-based heuristic
    demand_heuristic = demands / demands.sum()  # Normalize the demand vector
    
    # Combine the heuristics using a linear combination (weights can be adjusted)
    alpha = 0.5  # Weight for inverse distance heuristic
    beta = 0.5   # Weight for demand heuristic
    combined_heuristic = alpha * inverse_distance + beta * demand_heuristic
    
    # Subtracting from 1 to get negative values for undesirable edges
    return 1 - combined_heuristic