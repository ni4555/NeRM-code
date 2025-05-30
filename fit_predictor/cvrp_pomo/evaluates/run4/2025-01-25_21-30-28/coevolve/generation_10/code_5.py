import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Calculate the Normalization heuristic (sum of normalized demands at each node)
    normalization_heuristic = normalized_demands.sum(dim=1)
    
    # Calculate the Inverse Distance heuristic (inverse of the distance matrix)
    inverse_distance_heuristic = 1 / distance_matrix
    
    # Calculate the weighted sum of the heuristics
    # Weights can be adjusted as needed to favor one heuristic over the other
    alpha = 0.5  # Example weight for Normalization heuristic
    combined_heuristic = alpha * normalization_heuristic + (1 - alpha) * inverse_distance_heuristic
    
    # Ensure negative values for undesirable edges
    combined_heuristic[distance_matrix == 0] = 0  # Avoid division by zero
    combined_heuristic[combined_heuristic < 0] = 0  # Set undesirable edges to zero
    
    return combined_heuristic