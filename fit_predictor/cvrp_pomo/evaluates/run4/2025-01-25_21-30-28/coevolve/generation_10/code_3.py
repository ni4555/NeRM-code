import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity (sum of demands)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the distance matrix for use in the inverse distance heuristic
    # Note: We use the square root of the inverse distances to give more weight to shorter distances
    # and to ensure that the result is a proper distance matrix.
    # We also add a small epsilon to avoid division by zero.
    epsilon = 1e-6
    inv_distance_matrix = torch.sqrt(1 / (distance_matrix + epsilon))
    
    # Apply the Normalization heuristic by multiplying normalized demands by the distance matrix
    normalization_heuristic = normalized_demands * distance_matrix
    
    # Apply the Inverse Distance heuristic by multiplying the inverse distance matrix by the demands
    inverse_distance_heuristic = inv_distance_matrix * demands
    
    # Combine both heuristics to get a final heuristic matrix
    # The coefficients can be adjusted to balance the influence of each heuristic
    heuristic_matrix = 0.5 * (normalization_heuristic + inverse_distance_heuristic)
    
    return heuristic_matrix