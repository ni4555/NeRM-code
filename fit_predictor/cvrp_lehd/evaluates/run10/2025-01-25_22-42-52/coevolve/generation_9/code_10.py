import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the matrix to use as a base for negative heuristic values
    max_distance = torch.max(distance_matrix)
    
    # Calculate the sum of demands to normalize the demands vector
    sum_of_demands = torch.sum(demands)
    
    # Normalize demands
    normalized_demands = demands / sum_of_demands
    
    # Create a tensor of the same size as the distance matrix with all zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Loop over the distance matrix to compute the heuristics
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:  # Exclude the depot from the heuristics
                # Compute a simple heuristic value as a combination of distance and normalized demand
                heuristics[i, j] = -distance_matrix[i, j] - normalized_demands[j]
            else:  # For the depot node, assign a large negative value
                heuristics[i, j] = -max_distance
    
    return heuristics