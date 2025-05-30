import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load difference matrix
    load_diff = distance_matrix * demands
    
    # Normalize the load difference matrix by the maximum absolute load difference
    max_load_diff = torch.max(torch.abs(load_diff))
    load_diff_normalized = load_diff / max_load_diff
    
    # Calculate the cost matrix
    cost_matrix = distance_matrix + load_diff_normalized
    
    # Subtract the cost matrix from the maximum possible value to get the heuristics
    max_distance = torch.max(distance_matrix)
    heuristics = max_distance - cost_matrix
    
    return heuristics