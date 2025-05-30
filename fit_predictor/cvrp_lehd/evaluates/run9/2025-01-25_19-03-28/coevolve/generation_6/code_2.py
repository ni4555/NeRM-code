import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to represent the fraction of vehicle capacity needed by each customer
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the negative cost for each edge based on the normalized demand
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix
    
    # Add a small positive value to the diagonal to avoid selecting the depot as a customer
    heuristics += 1e-5 * torch.eye(distance_matrix.shape[0], device=distance_matrix.device)
    
    return heuristics