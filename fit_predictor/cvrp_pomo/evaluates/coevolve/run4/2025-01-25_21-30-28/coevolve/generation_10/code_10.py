import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device (CPU or GPU)
    distance_matrix = distance_matrix.to(demands.device)
    demands = demands.to(demands.device)

    # Step 1: Calculate the inverse distance heuristic
    # Inverse distance heuristic: the further the distance, the lower the heuristic value (negative)
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small epsilon to avoid division by zero

    # Step 2: Apply the Normalization heuristic
    # Calculate the total demand
    total_demand = demands.sum()
    # Normalize the distance matrix by the total demand
    normalized_distance = distance_matrix / total_demand

    # Step 3: Combine both heuristics to get the heuristic value for each edge
    # Negative value for inverse distance indicates a more promising edge
    # The normalization heuristic provides additional weight to edges that are more "loaded" (high demand)
    heuristics = -inverse_distance * normalized_distance

    # Step 4: Adjust the range of heuristic values to ensure non-negative entries
    # Heuristic values are clamped to be non-negative
    heuristics = torch.clamp(heuristics, min=0)

    return heuristics