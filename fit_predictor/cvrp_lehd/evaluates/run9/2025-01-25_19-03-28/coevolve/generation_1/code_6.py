import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized
    total_capacity = demands[0].item()  # Assuming the total vehicle capacity is the demand of the depot node
    normalized_demands = demands / total_capacity
    
    # Create a matrix of demand-to-capacity differences, scaled by distance
    # For each edge (i, j), the heuristics are determined by the normalized demand of customer j and the distance to j
    # We use negative demand to represent the cost of serving the customer
    heuristics_matrix = (normalized_demands - 1) * distance_matrix
    
    # We use a simple heuristic of taking the minimum of the negative heuristics to prioritize edges
    # This will give higher scores to edges with higher demand-to-capacity ratios (more promising)
    # The `torch.min` function is vectorized and efficient
    min_heuristics_matrix = torch.min(-heuristics_matrix, dim=0)[0]
    
    return min_heuristics_matrix