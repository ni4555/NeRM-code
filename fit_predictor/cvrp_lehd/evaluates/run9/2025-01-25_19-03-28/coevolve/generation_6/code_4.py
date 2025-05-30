import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix and demands tensor are on the same device
    demands = demands.to(distance_matrix.device)

    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values
    # We use a simple heuristic based on the difference in normalized demand
    # Promising edges are those with a demand closer to 1 (normalized demand)
    # Unpromising edges are those with a demand far from 1
    heuristics = (1 - torch.abs(normalized_demands - 1))

    return heuristics