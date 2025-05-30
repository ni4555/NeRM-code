import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the normalized demands as the ratio of each customer's demand to the total demand
    total_demand = demands.sum()
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values based on the normalized demands
    # Negative values for edges that are not promising
    # Positive values for edges that are promising
    # Here we use a simple heuristic based on the inverse of the demand
    heuristics = 1 / (normalized_demands.unsqueeze(1) + distance_matrix)
    
    # Replace inf values (from division by zero) with a very small number
    heuristics = torch.clamp(heuristics, min=1e-10)
    
    return heuristics