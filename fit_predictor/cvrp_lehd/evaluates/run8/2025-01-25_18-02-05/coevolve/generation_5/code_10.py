import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix so that the minimum distance is 0
    min_distance = distance_matrix.min()
    normalized_distance_matrix = distance_matrix - min_distance
    
    # Compute the heuristic values as a function of distance and demand
    # Negative values for undesirable edges and positive for promising ones
    # We can use a simple heuristic where the heuristic value is inversely proportional
    # to the distance and inversely proportional to the demand (to encourage picking up
    # customers with lower demands first)
    heuristics = -normalized_distance_matrix / (demands[:, None] + 1e-8)  # Adding a small constant to avoid division by zero
    
    return heuristics