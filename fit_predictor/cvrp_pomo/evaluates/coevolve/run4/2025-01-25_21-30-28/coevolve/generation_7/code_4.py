import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Inverse distance heuristic: closer nodes are more promising
    inv_distance_heuristic = -torch.log(distance_matrix)
    
    # Demand normalization heuristic: normalize demands to balance allocation
    normalized_demands = demands / demands.sum()
    
    # Enhance the heuristic with the demand information
    enhanced_heuristic = inv_distance_heuristic * normalized_demands
    
    # Ensure non-negative heuristics
    enhanced_heuristic = torch.clamp(enhanced_heuristic, min=0)
    
    return enhanced_heuristic