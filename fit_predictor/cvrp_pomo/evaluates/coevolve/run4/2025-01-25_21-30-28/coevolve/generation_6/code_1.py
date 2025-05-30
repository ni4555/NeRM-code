import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse of the demands to prioritize closer nodes
    inverse_demands = 1 / demands
    # Normalize the inverse demands by the sum of all demands to balance the importance
    normalized_inverse_demands = inverse_demands / inverse_demands.sum()
    # Use the normalized inverse demands to calculate the heuristics
    heuristics = normalized_inverse_demands * distance_matrix
    return heuristics