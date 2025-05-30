import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the size of the matrix
    n = distance_matrix.size(0)
    
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a matrix for normalized demand-based heuristic
    demand_based_heuristic = torch.zeros_like(distance_matrix)
    
    # Calculate negative demand-based heuristic
    for i in range(1, n):  # Exclude the depot
        for j in range(1, n):  # Exclude the depot
            demand_based_heuristic[i][j] = -normalized_demands[i] * normalized_demands[j]
    
    # Create a matrix for negative weighted distance heuristic
    distance_based_heuristic = -distance_matrix
    
    # Combine the two heuristics
    combined_heuristic = demand_based_heuristic + distance_based_heuristic
    
    # Introduce a slight perturbation to ensure the algorithm doesn't always pick the shortest paths
    combined_heuristic = combined_heuristic + torch.randn_like(combined_heuristic) * 0.1
    
    # Set the diagonal to negative infinity as no self-loops are desired
    torch.fill_diagonal_indices(combined_heuristic, float('-inf'))
    
    return combined_heuristic