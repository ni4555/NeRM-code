import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Calculate normalized distance matrix (using a small constant to avoid division by zero)
    norm_distance_matrix = distance_matrix.clone()
    min_distance = torch.min(distance_matrix, dim=1, keepdim=True)[0]
    norm_distance_matrix = (norm_distance_matrix - min_distance) / (torch.sum(distance_matrix, dim=1) - min_distance)
    
    # Calculate demand-based priority matrix (negative demands to give priority to lower demand)
    demand_priority_matrix = -demands / torch.sum(demands)
    
    # Combine the two matrices, giving higher priority to lower distances and demand
    combined_priority_matrix = norm_distance_matrix + demand_priority_matrix
    
    # Add a small constant to avoid issues with division by zero
    combined_priority_matrix += 1e-6
    
    # Normalize the combined priority matrix to ensure it is within the range [0, 1]
    max_priority = torch.max(combined_priority_matrix)
    min_priority = torch.min(combined_priority_matrix)
    priority_matrix = (combined_priority_matrix - min_priority) / (max_priority - min_priority)
    
    return priority_matrix