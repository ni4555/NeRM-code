import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_threshold = (demands / total_capacity).sum() / 2  # A simple heuristic for the threshold
    
    # Calculate the heuristics
    heuristics = -distance_matrix + demands * demand_threshold
    
    # Normalize the heuristics to ensure that all values are between -1 and 1
    min_val = heuristics.min()
    max_val = heuristics.max()
    heuristics = (heuristics - min_val) / (max_val - min_val)
    
    return heuristics