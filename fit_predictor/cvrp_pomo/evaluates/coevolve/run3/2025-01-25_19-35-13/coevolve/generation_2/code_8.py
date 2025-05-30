import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming all vehicles have the same capacity)
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics as a function of normalized demands and distance
    # Here we use a simple heuristic where the lower the distance, the higher the heuristic value
    # and we penalize nodes with higher normalized demand
    heuristics = -distance_matrix + normalized_demands
    
    return heuristics