import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the cumulative sum of normalized demands
    cumulative_demands = torch.cumsum(normalized_demands, dim=0)
    
    # Compute the heuristics based on the cumulative demands
    heuristics = distance_matrix - cumulative_demands.unsqueeze(1)
    
    # Ensure that the heuristics are negative for undesirable edges
    heuristics[distance_matrix == 0] = float('-inf')  # Avoid considering the depot node
    heuristics[heuristics < 0] = 0
    
    return heuristics