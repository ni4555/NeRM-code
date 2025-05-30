import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the negative of distances for heuristics
    # (we want to promote short distances as positive values)
    negative_distances = -distance_matrix
    
    # Combine normalized demands with the negative distances
    # We add demands to encourage selection of nodes with high demand
    heuristics = negative_distances + normalized_demands
    
    # Avoid promoting edges that lead to overflow by adding the total capacity
    # This ensures that edges with higher than total capacity are not selected
    heuristics += demands[:, None] - total_capacity
    
    return heuristics