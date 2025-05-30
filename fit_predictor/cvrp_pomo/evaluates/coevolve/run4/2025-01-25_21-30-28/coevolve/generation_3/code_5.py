import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands to get the load per unit distance
    load_per_unit_distance = demands / distance_matrix.sum()
    
    # Calculate the heuristic values
    # The heuristic value for each edge is the negative of the load per unit distance
    # since we want to minimize the load per unit distance
    heuristics = -load_per_unit_distance
    
    return heuristics