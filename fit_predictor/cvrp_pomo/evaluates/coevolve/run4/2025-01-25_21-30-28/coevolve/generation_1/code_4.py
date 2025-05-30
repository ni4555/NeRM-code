import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands to detect load imbalances
    demand_diff = demands - demands.mean()
    
    # Normalize the distance matrix to have a range of [0, 1]
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Use the demand difference to penalize edges that lead to imbalances
    demand_penalty = demand_diff.unsqueeze(1) * demand_diff.unsqueeze(0)
    
    # Combine the normalized distances and demand penalties
    combined_matrix = normalized_distance_matrix + demand_penalty
    
    # Apply a threshold to convert the combined matrix into a heuristics matrix
    # This threshold can be adjusted based on the specific problem context
    threshold = 0.5
    heuristics_matrix = torch.where(combined_matrix > threshold, combined_matrix, -combined_matrix)
    
    return heuristics_matrix