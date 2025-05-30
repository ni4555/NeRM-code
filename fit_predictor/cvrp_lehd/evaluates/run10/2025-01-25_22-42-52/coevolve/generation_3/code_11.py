import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential for each edge
    # Promising edges are those with high demand at the destination node
    potential_matrix = distance_matrix * normalized_demands
    
    # Introduce a negative value for the diagonal (depot to itself)
    # This prevents including the depot as a customer in the solution
    negative_diagonal = -1e5 * torch.eye(distance_matrix.size(0), dtype=potential_matrix.dtype)
    potential_matrix += negative_diagonal
    
    return potential_matrix