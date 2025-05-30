import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity
    
    # Calculate the sum of the demands for each node
    sum_demands = demands + demands[:, None]  # Add column-wise demands to itself for each row
    
    # Calculate the heuristic values for each edge
    # This heuristic is based on the difference in demand between the current node and the next node
    heuristics = (sum_demands - distance_matrix) * demands_normalized
    
    return heuristics