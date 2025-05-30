import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the demands to use them as a heuristic
    negative_demands = -demands
    
    # Use the demands as a heuristic value
    heuristics = negative_demands * (distance_matrix != 0)
    
    # Return the heuristics tensor
    return heuristics