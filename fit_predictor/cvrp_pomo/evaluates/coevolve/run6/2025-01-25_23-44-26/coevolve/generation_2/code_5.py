import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values based on distance and demand
    heuristics = -distance_matrix + normalized_demands
    
    # Normalize the heuristics to ensure all values are between -1 and 1
    min_val = heuristics.min()
    max_val = heuristics.max()
    heuristics = (heuristics - min_val) / (max_val - min_val)
    
    # Ensure the heuristics are within the range [-1, 1]
    heuristics = torch.clamp(heuristics, min=-1, max=1)
    
    return heuristics