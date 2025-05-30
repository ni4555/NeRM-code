import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative of the demand as a heuristic for undesirable edges
    # The more the demand, the less desirable the edge (negative heuristic)
    negative_heuristic = -normalized_demands.unsqueeze(0) * distance_matrix

    # Calculate the positive heuristic for promising edges
    # Subtract the negative heuristic from the distance matrix to get a positive heuristic
    positive_heuristic = distance_matrix - negative_heuristic

    # Return the heuristic matrix with negative values for undesirable edges
    # and positive values for promising ones
    return positive_heuristic