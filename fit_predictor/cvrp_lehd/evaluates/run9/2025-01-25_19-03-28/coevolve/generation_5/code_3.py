import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    
    # Normalize demands to be in the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Calculate the heuristic values using the triangle inequality
    # and the demand of each customer node
    heuristics = distance_matrix - torch.clamp(distance_matrix, min=0) + \
                  torch.clamp(normalized_demands.unsqueeze(0).expand(n, n), max=0)
    
    # Adjust the heuristics to ensure the inclusion of edges
    # with non-zero demand
    heuristics[torch.arange(n), torch.arange(n)] = -float('inf')
    heuristics[torch.arange(n), demands > 0] = 0  # Ensure edges to customers with demand
    
    return heuristics