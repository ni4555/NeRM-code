import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()  # Assuming vehicle capacity is the sum of all demands
    
    # Normalize demands relative to vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the sum of normalized demands for each edge
    sum_normalized_demands = torch.matmul(normalized_demands, normalized_demands.t())
    
    # Incorporate distance and road quality factors
    # Assuming distance_matrix has been scaled to include road quality factors
    # For simplicity, let's assume road quality is represented by a factor in the distance_matrix
    edge_potential = sum_normalized_demands - distance_matrix
    
    # Introduce a small constant to prevent division by zero errors
    epsilon = 1e-8
    edge_potential = edge_potential.clamp(min=epsilon)
    
    # Apply a robust potential function to assign weights
    # For example, a simple function could be the negative of the potential function
    weights = -edge_potential
    
    return weights