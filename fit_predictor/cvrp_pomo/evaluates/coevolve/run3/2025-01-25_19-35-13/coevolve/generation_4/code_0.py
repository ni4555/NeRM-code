import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_vector = demands / total_capacity
    
    # Calculate the heuristic for each edge based on the demands and distances
    heuristics = (1 / (distance_matrix + 1e-5)) * (1 - demand_vector)
    
    return heuristics