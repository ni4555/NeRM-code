import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Define a demand threshold (for demonstration purposes, we set it to 0.5).
    # This threshold can be adjusted based on the problem specifics.
    demand_threshold = 0.5
    
    # Normalize the distance matrix by the maximum distance in the matrix
    # to prevent any distance value from dominating the computation.
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the product of normalized distances and demands to find promising edges.
    demand_normalized_distance = demands * normalized_distance_matrix
    
    # Assign negative values to edges where the total demand exceeds the threshold.
    # The penalty factor is chosen arbitrarily for demonstration purposes.
    penalty_factor = -100
    demand_exceeds_threshold = demand_normalized_distance > demand_threshold
    penalty_matrix = penalty_factor * demand_exceeds_threshold
    
    # Apply the penalty to the demand normalized distance matrix.
    heuristics_matrix = demand_normalized_distance + penalty_matrix
    
    # Add a small positive constant to ensure that no edge has a negative heuristic value.
    epsilon = 1e-5
    heuristics_matrix = torch.clamp(heuristics_matrix, min=epsilon)
    
    return heuristics_matrix