import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands from the normalized demand
    demand_diff = demands - demands.mean()
    
    # Calculate the negative of the absolute difference in demands
    # This will encourage edges that lead to a more balanced load
    demand_diff_neg = -torch.abs(demand_diff)
    
    # Calculate the sum of the absolute differences in distances
    # This will encourage edges that are shorter
    distance_diff = torch.abs(distance_matrix - distance_matrix.mean(axis=0, keepdim=True))
    
    # Combine the two heuristics using a simple linear combination
    # The coefficients (alpha and beta) can be adjusted to emphasize certain criteria
    alpha, beta = 0.5, 0.5
    heuristic_matrix = alpha * demand_diff_neg + beta * distance_diff
    
    # Ensure that the heuristic matrix is of the same shape as the distance matrix
    assert heuristic_matrix.shape == distance_matrix.shape, "Heuristic matrix shape does not match distance matrix shape."
    
    return heuristic_matrix