import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1, assuming the total capacity is 1
    normalized_demands = demands / demands.sum()
    
    # Create a matrix of negative infinity to represent undesirable edges
    heuristics = -torch.ones_like(distance_matrix)
    
    # Iterate through each customer
    for i in range(1, distance_matrix.shape[0]):
        # Add the normalized demand of the current customer to the edge weight
        heuristics[i, :] = distance_matrix[i, :] - normalized_demands[i]
    
    # Ensure the diagonal elements are zero (self-loops are not desirable)
    torch.fill_diagonal(heuristics, 0)
    
    return heuristics