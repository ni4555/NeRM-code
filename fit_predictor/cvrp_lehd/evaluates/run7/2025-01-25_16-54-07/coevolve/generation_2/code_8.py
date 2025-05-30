import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the sum of normalized demands along the diagonal to avoid self-assignment
    sum_normalized_demands = normalized_demands.sum(dim=1, keepdim=True)
    
    # Subtract the sum of normalized demands from each demand to get the relative contribution
    relative_contributions = demands - sum_normalized_demands
    
    # Use the relative contributions as the heuristic value for each edge
    # We use negative values for undesirable edges and positive values for promising ones
    # For simplicity, we use the negative of the relative contributions
    heuristics = -relative_contributions
    
    return heuristics