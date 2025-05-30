import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the cumulative demand mask
    cumulative_demand_mask = torch.cumsum(demands, dim=0)
    
    # Calculate the normalized demand
    normalized_demand = (demands - cumulative_demand_mask[:-1]) / cumulative_demand_mask[1:]
    
    # Compute the heuristic values
    # The heuristic is designed to be positive where the demand is high and distance is low
    # and negative where the demand is low or distance is high.
    heuristics = -distance_matrix + normalized_demand
    
    # Ensure the heuristics matrix has the same shape as the distance matrix
    assert heuristics.shape == distance_matrix.shape, "Heuristics shape does not match distance matrix shape."
    
    return heuristics