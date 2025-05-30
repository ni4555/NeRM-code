import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity by summing the demands
    total_capacity = demands.sum()
    
    # Normalize demands to represent fractions of the total capacity
    normalized_demands = demands / total_capacity
    
    # Initialize a tensor with the same shape as the distance matrix to store heuristics
    heuristics = torch.full_like(distance_matrix, fill_value=-1e9)
    
    # Iterate over each node pair to calculate heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Exclude the depot node from comparisons
                # Calculate the heuristic for the edge (i, j)
                heuristic = distance_matrix[i, j] - normalized_demands[i] * normalized_demands[j]
                heuristics[i, j] = heuristic
    
    return heuristics