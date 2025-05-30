import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix and demands are already normalized as described.
    
    # Calculate the maximum distance in the matrix to normalize heuristics
    max_distance = distance_matrix.max()
    
    # Normalize demands to ensure they don't exceed the vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristic value for each edge based on the distance and demand
    # The heuristic function used here is a simple combination of distance and demand
    # where the demand is adjusted to ensure positive values.
    heuristics = -distance_matrix + (normalized_demands * 10)  # Scale demand coefficient as needed
    
    # Replace any negative values with 0 as they are undesirable edges
    heuristics[heuristics < 0] = 0
    
    return heuristics