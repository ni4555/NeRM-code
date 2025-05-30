import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values as a function of demand and distance
    # The heuristic function is an example; it should be designed based on the specific problem and requirements
    heuristics = -normalized_demands * distance_matrix
    
    return heuristics