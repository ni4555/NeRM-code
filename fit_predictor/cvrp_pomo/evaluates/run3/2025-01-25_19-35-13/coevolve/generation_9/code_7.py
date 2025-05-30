import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristics matrix with zeros
    n = distance_matrix.shape[0]
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values based on customer demands
    # We use the negative demand to create a heuristic that prioritizes nodes with lower demands
    heuristics[1:] = -demands[1:]

    # Incorporate distance penalties to avoid long distances
    # We can use a simple function that increases the penalty with distance
    heuristics += torch.clamp(distance_matrix[1:], min=0)

    # Normalize the heuristics matrix to ensure that the sum of heuristics for each row (customer) is zero
    # This ensures that the total distance covered for each customer is zero
    total_heuristics = heuristics.sum(dim=1)
    heuristics /= (total_heuristics.clamp(min=1) + 1e-10)  # Add a small value to avoid division by zero

    return heuristics