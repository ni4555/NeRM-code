import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros to store the heuristics values
    heuristics = torch.zeros_like(distance_matrix)
    
    # Demand relaxation: Calculate the total normalized demand
    total_demand = demands.sum()
    
    # Node partitioning: For each customer, find the nearest depot
    # This is a placeholder for a proper node partitioning algorithm
    nearest_depot_indices = torch.argmin(distance_matrix[:, 1:], dim=1) + 1
    
    # Calculate the distance from each customer to its nearest depot
    distances_to_nearest_depot = distance_matrix[nearest_depot_indices, 0]
    
    # Path decomposition: Calculate the demand of the path from the depot to the nearest depot
    path_demand = demands[nearest_depot_indices]
    
    # Explicit potential value heuristic: Calculate the potential value for each edge
    for i in range(1, n):  # Skip the depot node
        for j in range(1, n):  # Skip the depot node
            # Calculate the potential value of including the edge from i to j
            # This is a simplified heuristic that considers the distance and demand
            potential_value = distance_matrix[i, j] - (demands[j] / total_demand) * distances_to_nearest_depot[i]
            heuristics[i, j] = potential_value
    
    return heuristics