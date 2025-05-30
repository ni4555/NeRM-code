import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total vehicle capacity (for normalization)
    vehicle_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Apply Inverse Distance Heuristic (IDH) for initial customer assignment
    # Promising edges have negative values, undesirable edges have positive values
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the inverse distance
                inverse_distance = 1 / distance_matrix[i, j]
                # Add demand-driven weight
                demand_weight = normalized_demands[i] * normalized_demands[j]
                # Apply capacity constraint penalty
                capacity_penalty = torch.abs(1 - (demands[i] + demands[j]) / vehicle_capacity)
                # Combine the heuristics
                heuristic_matrix[i, j] = -inverse_distance * demand_weight + capacity_penalty
    
    # Add a small constant to avoid division by zero in log-sum-exp
    epsilon = 1e-10
    heuristic_matrix = heuristic_matrix + epsilon
    
    # Normalize the heuristic matrix to ensure all values are between 0 and 1
    max_value = torch.max(heuristic_matrix)
    min_value = torch.min(heuristic_matrix)
    heuristic_matrix = (heuristic_matrix - min_value) / (max_value - min_value)
    
    return heuristic_matrix