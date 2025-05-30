import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands, excluding the depot demand
    total_capacity = demands.sum() - demands[0]
    
    # Normalize demands by total capacity
    normalized_demands = demands / total_capacity
    
    # Compute a simple heuristic based on the normalized demand of each node
    # The heuristic for each edge (i, j) is the negative of the normalized demand of the customer at node j
    # This encourages routes that visit customers with higher demand earlier
    heuristics = -normalized_demands[1:] * distance_matrix[1:, 1:]
    
    # The heuristics for the edges leading from the depot to customers and customers to the depot are set to a high negative value
    # to discourage such edges in the heuristic evaluation
    heuristics[0, 1:] = heuristics[1:, 0] = -float('inf')
    
    return heuristics