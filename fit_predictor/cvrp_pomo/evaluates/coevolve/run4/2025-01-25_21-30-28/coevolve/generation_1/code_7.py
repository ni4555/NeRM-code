import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values
    # We use the formula: heuristics[i, j] = distance[i, j] * (1 - demands[i] * normalized_demands[j])
    # This heuristic encourages selecting edges with lower distance and higher demand match
    heuristics = distance_matrix * (1 - demands[:, None] * normalized_demands[None, :])
    
    return heuristics