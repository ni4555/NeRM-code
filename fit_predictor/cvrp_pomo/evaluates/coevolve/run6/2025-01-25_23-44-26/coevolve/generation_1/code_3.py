import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the sum of demands
    total_demand = demands.sum()
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    # For each customer, calculate the heuristic based on the normalized demand
    for i in range(1, n):  # Skip the depot node
        heuristics[i] = normalized_demands[i] * (distance_matrix[i, 0] + distance_matrix[i, 0])  # Example heuristic
    return heuristics