import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if demands are normalized by total vehicle capacity
    total_capacity = demands.sum()
    demands = demands / total_capacity
    
    # Calculate the heuristic values for each edge
    # The heuristic is based on the following formula:
    # heuristic[i][j] = demand[i] * (1 + demand[j])
    # where demand[i] is the normalized demand of customer i
    # This formula promotes selecting edges that have lower demand or are serving more customers
    heuristics = demands[:, None] * (1 + demands[None, :])
    
    # We add a small constant to avoid division by zero
    # and to ensure the heuristic is positive
    epsilon = 1e-10
    heuristics += epsilon
    
    # Calculate the sum of demands along each row and subtract from the heuristic
    # This encourages selecting edges that lead to balanced vehicle loads
    row_sums = heuristics.sum(dim=1, keepdim=True)
    heuristics -= row_sums
    
    # Subtract the distance matrix from the heuristic values
    # This penalizes longer distances, making them less likely to be chosen
    heuristics -= distance_matrix
    
    return heuristics