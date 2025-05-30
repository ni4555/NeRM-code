import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristic value for each edge
    # The heuristic is a combination of the normalized demand of the customer and the inverse of the distance
    # Negative values for undesirable edges and positive values for promising ones
    heuristics = -normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0) + 1 / distance_matrix
    
    return heuristics