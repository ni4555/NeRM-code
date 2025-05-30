import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demands_normalized = demands / vehicle_capacity
    
    # Calculate the cost of each edge as the negative of the demand normalized by the vehicle capacity
    # This heuristic will favor edges with lower normalized demand
    cost_matrix = -demands_normalized
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    cost_matrix = cost_matrix + epsilon
    
    # Normalize the cost matrix to have a range of values that are more suitable for PSO
    min_cost = cost_matrix.min()
    max_cost = cost_matrix.max()
    normalized_cost_matrix = (cost_matrix - min_cost) / (max_cost - min_cost)
    
    return normalized_cost_matrix