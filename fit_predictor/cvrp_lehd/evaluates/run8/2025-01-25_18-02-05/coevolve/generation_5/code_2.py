import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    # Normalize the demands
    normalized_demands = demands / total_demand
    # Calculate the heuristics matrix
    heuristics_matrix = -distance_matrix + (normalized_demands * distance_matrix)
    return heuristics_matrix