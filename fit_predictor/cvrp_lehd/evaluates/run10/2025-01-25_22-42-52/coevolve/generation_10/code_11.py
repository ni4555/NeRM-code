import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands
    normalized_demands = demands / demands.sum()
    # Calculate potential negative value for each edge
    negative_value = -distance_matrix
    # Calculate potential positive value based on demands
    positive_value = (1 - normalized_demands) * distance_matrix
    # Sum the two potential values to get the heuristic value
    heuristic_values = negative_value + positive_value
    return heuristic_values