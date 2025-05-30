import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total capacity for a simple normalization approach
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the negative distance to discourage longer paths
    negative_distance = -distance_matrix
    
    # Calculate the heuristics as a weighted sum of normalized demands and negative distance
    heuristics = (negative_distance + normalized_demands) * 0.5
    
    # Optionally, you could introduce additional heuristics based on specific problem characteristics
    
    return heuristics