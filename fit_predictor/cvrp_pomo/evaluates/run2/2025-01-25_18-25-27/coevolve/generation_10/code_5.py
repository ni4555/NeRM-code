import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic value for each edge
    # The heuristic is the negative of the demand (promising to include edges with higher demand)
    # and the distance (undesirable to include edges with higher distance)
    heuristics = -normalized_demands * distance_matrix
    
    # Normalize the heuristics to ensure that they are within a certain range
    # This step can be omitted if not required
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    
    return heuristics