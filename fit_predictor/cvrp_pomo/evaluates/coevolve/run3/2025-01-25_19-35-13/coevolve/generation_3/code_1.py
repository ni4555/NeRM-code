import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    distance_matrix /= torch.max(distance_matrix, dim=0).values[:, None]
    
    # Normalize the demands by the vehicle capacity (assuming 1 for simplicity)
    demands /= torch.sum(demands)
    
    # Create a demand matrix
    demand_matrix = torch.zeros_like(distance_matrix)
    demand_matrix[1:, :] = demands
    
    # Compute the heuristic based on both distance and demand
    heuristic_matrix = -distance_matrix + demand_matrix
    
    # Avoid division by zero errors by adding a small epsilon
    epsilon = 1e-10
    heuristic_matrix += epsilon
    
    return heuristic_matrix