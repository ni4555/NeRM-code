import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix to use as a heuristic
    # Negative values are more desirable in this context
    negative_distance_matrix = -distance_matrix
    
    # Normalize the negative distance matrix by the demands to get the heuristics
    # This encourages choosing edges that lead to nodes with lower demands
    heuristics = negative_distance_matrix / demands
    
    # Replace division by zero with a very small value to avoid NaNs
    heuristics = torch.where(demands == 0, torch.tensor(1e-8), heuristics)
    
    return heuristics