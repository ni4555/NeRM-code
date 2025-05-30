import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative weighted distance matrix
    negative_weighted_distance = -distance_matrix
    
    # Normalize the negative weighted distance matrix by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_distance_matrix = negative_weighted_distance / total_capacity
    
    # Calculate the demand-based heuristic
    demand_heuristic = demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # Combine the normalized distance and demand-based heuristic
    combined_heuristic = normalized_distance_matrix + demand_heuristic
    
    # Apply dynamic load balancing by prioritizing edges with lower combined heuristic values
    # This is a simple approach to simulate dynamic load balancing without complex calculations
    combined_heuristic = combined_heuristic.clamp(min=0)  # Ensure non-negative values
    
    return combined_heuristic