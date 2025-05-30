import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the inverse of the demand, which will be used to penalize high-demand edges
    demand_inverse = 1 / (demands + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Create a matrix to hold the heuristic values
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Assign a negative heuristic value to high-demand edges
    heuristic_matrix[demands > total_capacity / len(demands)] = -1
    
    # Normalize the distance matrix by the total capacity to account for varying problem scales
    normalized_distance_matrix = distance_matrix / total_capacity
    
    # Adjust the heuristic values based on the normalized distances
    heuristic_matrix += normalized_distance_matrix
    
    return heuristic_matrix