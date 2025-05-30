import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to represent fractions of the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Normalize distances to be in the same scale as demands
    # This can be a simple min-max scaling or another normalization technique
    min_distance = distance_matrix.min().item()
    max_distance = distance_matrix.max().item()
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate the potential value for explicit depot handling
    # This could be a fixed value or a value based on the maximum demand
    depot_potential = max(demands)
    
    # Initialize the heuristics matrix with negative values for all edges
    heuristics_matrix = -torch.ones_like(distance_matrix)
    
    # Loop over the nodes to calculate the heuristics
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            # For depot node 0, add the potential value of the depot
            if i == 0:
                heuristics_matrix[i, j] = depot_potential + normalized_distances[i, j] * normalized_demands[j]
            else:
                # For other nodes, add the distance and demand normalization to the heuristics
                heuristics_matrix[i, j] = normalized_distances[i, j] * normalized_demands[j]
    
    return heuristics_matrix