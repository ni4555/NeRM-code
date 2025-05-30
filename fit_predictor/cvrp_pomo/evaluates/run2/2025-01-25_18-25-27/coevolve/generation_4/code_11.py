import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    n = distance_matrix.shape[0]
    heuristics_matrix = torch.zeros(n, n, dtype=torch.float32)

    # Calculate the heuristics for each edge
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip the diagonal (no self-loops)
                # Calculate the heuristic based on normalized demand and distance
                # Here we are using a simple heuristic where we multiply the normalized demand by the distance
                heuristics_matrix[i, j] = normalized_demands[i] * distance_matrix[i, j]

    return heuristics_matrix