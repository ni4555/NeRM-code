import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    total_capacity = demands.sum()
    demands = demands / total_capacity
    
    # Calculate savings for each edge (i, j)
    savings_matrix = (distance_matrix * demands[:, None] * demands[None, :]).clamp(min=0)
    savings_matrix = savings_matrix.sum(dim=1) - distance_matrix.sum(dim=1)
    
    # Calculate the cost matrix which is the negative of savings
    cost_matrix = -savings_matrix
    
    # Subtract the savings of the depot node from the cost matrix
    cost_matrix[:, 0] -= savings_matrix[:, 0]
    
    # Add the cost of the depot node to the cost matrix
    cost_matrix[0, :] -= savings_matrix[0, :]
    
    return cost_matrix