import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a tensor with all values initialized to zero
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the cost for each edge as the sum of the normalized demand and the distance
    # Note: torch.sum returns the sum of the elements across a given dimension of the tensor
    # Here we sum across the rows, which correspond to edges
    heuristics_matrix = normalized_demands.unsqueeze(1) + distance_matrix
    
    return heuristics_matrix