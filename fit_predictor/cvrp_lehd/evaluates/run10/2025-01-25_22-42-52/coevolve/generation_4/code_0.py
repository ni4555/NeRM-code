import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix and demands are 1D tensors and demands are normalized
    n = distance_matrix.size(0)
    demands = demands / demands.sum()  # Normalize demands

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the sum of demands for all edges from each customer to the depot
    sum_of_demands = torch.sum(demands * distance_matrix)

    # Calculate the heuristic for each edge based on the normalized demand and distance
    # Negative values for undesirable edges (large distances), positive for promising ones
    heuristic_matrix = -distance_matrix + sum_of_demands

    # Ensure that diagonal elements (edges to the depot from the depot) are zero
    # as they do not contribute to the cost or demand
    torch.fill_diagonal(heuristic_matrix, 0)

    return heuristic_matrix