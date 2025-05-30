import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the inverse of normalized demands (heuristic value)
    # Larger distances should be considered as more desirable to include in the solution
    heuristic_matrix = 1 / (distance_matrix + 1e-6)  # Add a small constant to avoid division by zero
    
    # Scale the heuristic values based on the demands of each customer
    scaled_heuristic_matrix = heuristic_matrix * normalized_demands.unsqueeze(1)
    
    # Compute the negative sum of the scaled heuristic values along the rows (customers)
    # This encourages paths that serve customers with higher demand earlier
    negative_scaled_sum = -scaled_heuristic_matrix.sum(dim=1)
    
    return negative_scaled_sum