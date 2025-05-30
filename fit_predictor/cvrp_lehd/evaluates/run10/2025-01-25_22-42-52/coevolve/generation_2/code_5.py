import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of distances for each row (from depot to all other nodes)
    row_sums = distance_matrix.sum(dim=1)
    
    # Calculate the sum of distances for each column (from all nodes to depot)
    col_sums = distance_matrix.sum(dim=0)
    
    # Calculate the potential benefit of including each edge
    # The heuristic is a combination of the demand and the distance, adjusted by the normalized demand
    heuristics = row_sums + col_sums - 2 * distance_matrix * normalized_demands
    
    return heuristics