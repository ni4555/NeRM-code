import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized
    demands = demands / demands.sum()

    # Calculate the negative of the demands for the heuristic (undesirable edges)
    negative_demands = -demands

    # Compute the heuristic values by subtracting the negative demands from the distance matrix
    # This gives a higher score to edges with lower distances and lower demands (undesirable edges)
    heuristics = distance_matrix + negative_demands

    # Since we want negative values for undesirable edges and positive for promising ones,
    # we take the absolute value to ensure the heuristic is non-negative
    heuristics = torch.abs(heuristics)

    return heuristics