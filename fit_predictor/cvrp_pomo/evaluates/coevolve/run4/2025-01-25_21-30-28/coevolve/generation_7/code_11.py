import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic (IDH)
    idh = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate the demand normalization heuristic (DNH)
    dnh = normalized_demands

    # Combine IDH and DNH to get the heuristic values
    heuristics = idh * dnh

    return heuristics