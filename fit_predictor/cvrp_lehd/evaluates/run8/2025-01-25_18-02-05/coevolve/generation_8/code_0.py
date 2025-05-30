import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance_matrix is 2D
    if len(distance_matrix.shape) != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix (n by n).")
    
    # Ensure demands are a 1D tensor
    if len(demands.shape) != 1:
        raise ValueError("demands must be a 1D tensor.")
    
    # Get the number of customers (excluding the depot)
    num_customers = distance_matrix.shape[0] - 1
    
    # Calculate the sum of normalized demands (excluding the depot)
    sum_normalized_demands = torch.sum(demands[1:])
    
    # Calculate the heuristic values
    # A simple heuristic could be the negative of the distance to the depot plus a small value if demand is high
    heuristics = -distance_matrix[1:, 0] + demands[1:] * 0.1 * (1 / (1 + sum_normalized_demands))
    
    return heuristics