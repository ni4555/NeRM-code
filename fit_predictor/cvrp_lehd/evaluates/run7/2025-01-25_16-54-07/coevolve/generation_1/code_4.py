import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential benefit of including each edge in the solution
    # The benefit is higher for edges that have a lower distance and higher demand
    benefit = -distance_matrix + normalized_demands
    
    # Ensure all values are in the range of negative infinity to positive infinity
    benefit = torch.clamp(benefit, min=float('-inf'), max=float('inf'))
    
    return benefit