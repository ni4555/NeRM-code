import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the sum of distances for each row (from the depot to each customer)
    row_sums = distance_matrix.sum(dim=1)
    
    # Calculate the sum of distances for each column (from each customer to the depot)
    col_sums = distance_matrix.sum(dim=0)
    
    # Calculate the heuristic values
    heuristics = -((row_sums * normalized_demands.unsqueeze(1)).unsqueeze(0) +
                   (col_sums * normalized_demands.unsqueeze(0)))
    
    return heuristics