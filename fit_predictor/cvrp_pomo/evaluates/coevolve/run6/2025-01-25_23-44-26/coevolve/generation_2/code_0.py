import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values
    # For simplicity, we use the normalized demand as the heuristic value
    # This is a naive heuristic that could be improved with more sophisticated methods
    heuristic_matrix = normalized_demands[None, :] * normalized_demands[:, None]
    
    # Discourage long distances by subtracting them from the heuristic values
    heuristic_matrix -= distance_matrix
    
    # Introduce a penalty for high demand
    penalty = torch.max(torch.abs(heuristic_matrix), dim=1)[0].max()
    heuristic_matrix -= penalty[None, :]
    
    return heuristic_matrix