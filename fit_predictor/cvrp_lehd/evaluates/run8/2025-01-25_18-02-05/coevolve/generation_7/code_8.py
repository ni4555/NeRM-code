import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Initialize a tensor to store the heuristic values with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Compute the heuristic values using the formula:
    # heuristics[i][j] = distance_matrix[i][j] - distance_matrix[i][0] - distance_matrix[0][j] + normalized_demands[i] + normalized_demands[j]
    # where i and j are indices of the customers, and 0 is the depot index.
    heuristics = distance_matrix - distance_matrix[:, None, :] - distance_matrix[None, :, :] + normalized_demands[:, None] + normalized_demands[None, :]
    
    return heuristics