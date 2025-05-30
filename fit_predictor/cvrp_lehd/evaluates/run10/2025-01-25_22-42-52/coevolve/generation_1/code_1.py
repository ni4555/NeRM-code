import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are of compatible shapes
    if distance_matrix.shape != (demands.shape[0], demands.shape[0]):
        raise ValueError("The distance matrix must have the same number of rows and columns as the number of nodes.")

    # Initialize a matrix of the same shape as distance_matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the negative demand difference for each edge (promising edges have negative differences)
    demand_difference = -demands[:, None] + demands[None, :]

    # Incorporate the distance into the heuristic, prioritizing closer nodes
    heuristics_matrix = distance_matrix + demand_difference

    # Add a small positive value to avoid negative infinity in case of zero demand difference
    heuristics_matrix = torch.clamp(heuristics_matrix, min=1e-8)

    return heuristics_matrix