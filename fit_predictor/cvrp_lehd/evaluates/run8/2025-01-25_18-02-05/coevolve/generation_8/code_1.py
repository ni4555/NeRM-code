import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the input tensors are on the same device (e.g., GPU if available)
    # This is necessary for vectorized operations
    distance_matrix = distance_matrix.to(demands.device)
    demands = demands.to(demands.device)

    # Normalize the demands vector by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize a tensor of the same shape as distance_matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Compute the heuristics: the negative of the product of the distance and the normalized demand
    # This heuristic assumes that shorter distances and lower demands are more promising
    heuristics = -distance_matrix * normalized_demands

    return heuristics