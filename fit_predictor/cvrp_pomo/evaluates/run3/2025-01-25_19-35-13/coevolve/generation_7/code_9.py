import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each edge
    sum_normalized_demands = torch.clamp(normalized_demands[:, None] + normalized_demands[None, :], min=1e-8)
    
    # Incorporate distance and road quality factors (assuming these are provided as additional tensors)
    distance_factor = distance_matrix
    road_quality_factor = torch.rand_like(distance_matrix)  # Placeholder for road quality factor
    
    # Combine factors to create the potential function
    potential_function = sum_normalized_demands * distance_factor * road_quality_factor
    
    # Introduce a heuristic for the edges (negative values for undesirable edges)
    heuristics = -potential_function
    
    return heuristics