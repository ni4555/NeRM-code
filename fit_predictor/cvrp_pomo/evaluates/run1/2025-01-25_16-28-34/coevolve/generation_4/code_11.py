import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Greedy assignment based on demand
    for i in range(1, n):
        min_distance = normalized_distance_matrix[0, i:].min()
        min_index = torch.where(normalized_distance_matrix[0, i:] == min_distance)[0].item()
        heuristics[0, i] = normalized_demands[i] * min_distance
        normalized_distance_matrix[:, i] += distance_matrix[0, i]  # Exclude the depot node for the next iteration
    
    return heuristics