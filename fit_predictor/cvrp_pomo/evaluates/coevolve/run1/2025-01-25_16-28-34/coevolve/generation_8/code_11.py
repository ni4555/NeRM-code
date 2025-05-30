import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    # Normalize demands
    normalized_demands = demands / total_demand

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cost of each edge based on normalized demand
    # We use a simple heuristic where the cost is inversely proportional to the demand
    heuristics = 1 / (normalized_demands + 1e-8)  # Adding a small constant to avoid division by zero

    # Apply a penalty for edges leading to the depot (index 0)
    heuristics[torch.arange(n), 0] *= -1

    return heuristics