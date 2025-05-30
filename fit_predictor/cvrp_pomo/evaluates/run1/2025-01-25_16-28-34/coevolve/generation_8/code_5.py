import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Calculate the maximum distance between any two customers
    max_distance = distance_matrix.max()
    
    # Calculate the total demand as a fraction of the vehicle capacity
    demand_fraction = demands / vehicle_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # For each edge in the distance matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal (self-loops)
                # Calculate the heuristic value based on distance and demand
                # Negative values for undesirable edges, positive for promising ones
                heuristic_value = -distance_matrix[i, j] + max_distance * demand_fraction[j]
                # Update the heuristic matrix
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix