import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Compute the heuristics
    # We can use a simple heuristic such as the inverse of the distance (assuming that shorter distances are more promising)
    # and adjust it by the normalized demand
    heuristics = (1 / distance_matrix) * normalized_demands
    
    # Ensure that the values are within a certain range to avoid numerical issues
    heuristics = torch.clamp(heuristics, min=-1e10, max=1e10)
    
    return heuristics