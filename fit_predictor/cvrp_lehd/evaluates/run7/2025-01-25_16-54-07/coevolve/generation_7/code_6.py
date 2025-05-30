import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate cumulative demand mask
    cumulative_demand_mask = (demands > 0).float()
    
    # Calculate cumulative demand for each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Normalize cumulative demand
    normalized_demand = cumulative_demand / (cumulative_demand[-1] + 1e-6)
    
    # Compute the heuristic value for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Compute heuristic based on distance and normalized demand
                heuristic = -distance_matrix[i, j] + normalized_demand[j]
                # Adjust heuristic for the cumulative demand mask
                heuristic += (cumulative_demand_mask * demands[j])
                # Apply the heuristic to the matrix
                heuristic_matrix[i, j] = heuristic
    
    return heuristic_matrix