import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    demands_sum = demands.sum()
    normalized_demands = demands / demands_sum

    # Calculate the heuristic for each edge based on distance and normalized demand
    # The heuristic is designed to be higher for edges that are short and have higher demands
    # This is a simple heuristic example that might be adapted to more complex ones as needed
    heuristic_matrix = -distance_matrix + (normalized_demands.unsqueeze(0) * demands.unsqueeze(1))

    return heuristic_matrix