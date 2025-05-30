import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Calculate the cumulative demand matrix
    cumulative_demand_matrix = torch.cumsum(demands, dim=0)
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Apply problem-specific local search
    # Example: Use a simple greedy approach to calculate heuristic values
    for i in range(1, len(demands) + 1):
        for j in range(1, i + 1):
            # Calculate the load after including this edge
            current_load = cumulative_demand_matrix[j - 1] - cumulative_demand_matrix[i - 1]
            # Calculate the heuristic value based on load
            heuristic_matrix[i, j] = -current_load if current_load > 1.0 else normalized_distance_matrix[i, j]
    
    return heuristic_matrix