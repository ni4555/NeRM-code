import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix so that the diagonal is 0 and all other entries are non-negative
    distance_matrix = distance_matrix.clamp(min=0)
    identity = torch.eye(distance_matrix.size(0), dtype=distance_matrix.dtype, device=distance_matrix.device)
    distance_matrix -= identity  # Subtract the identity matrix to get the distance between nodes

    # Calculate the cumulative demand matrix
    cumulative_demand = torch.cumsum(demands[:, None], dim=1).squeeze(1)

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # For each vehicle, calculate the heuristic value
    for i in range(demands.size(0)):  # For each customer node
        # Calculate the heuristic value for the current customer node
        heuristic_matrix[i] = -distance_matrix[i] * cumulative_demand

    # Adjust the heuristic matrix to ensure non-negative values
    # The adjustment factor is the minimum heuristic value across all nodes
    adjustment_factor = torch.min(heuristic_matrix)
    if adjustment_factor < 0:
        heuristic_matrix += -adjustment_factor

    return heuristic_matrix