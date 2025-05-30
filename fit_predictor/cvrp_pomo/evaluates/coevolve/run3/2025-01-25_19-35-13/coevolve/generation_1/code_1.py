import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total capacity
    total_capacity = demands.sum()

    # Normalize the demands to get the demand per unit capacity
    normalized_demands = demands / total_capacity

    # Compute the heuristic value for each edge
    # The heuristic is a combination of the demand ratio and the distance
    # The formula used here is a simple linear combination: h = demand_ratio * distance
    # Negative values are assigned to edges with high demand per unit distance
    heuristic_matrix = normalized_demands.unsqueeze(1) * distance_matrix

    # To ensure the heuristic matrix has negative values for undesirable edges and
    # positive values for promising ones, we can adjust the values by subtracting the
    # minimum value in the heuristic matrix from all elements, then adding a small positive
    # value to all to avoid zeros
    min_heuristic = heuristic_matrix.min()
    adjusted_heuristic_matrix = heuristic_matrix - min_heuristic
    adjusted_heuristic_matrix = adjusted_heuristic_matrix + min(min_heuristic, 1e-5)

    return adjusted_heuristic_matrix