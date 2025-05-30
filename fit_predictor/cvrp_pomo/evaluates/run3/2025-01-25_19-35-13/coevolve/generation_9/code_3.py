import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demands vector
    total_demand = demands.sum()
    
    # Normalize the demands vector
    normalized_demands = demands / total_demand
    
    # Calculate the sum of demands for each row (customer node)
    row_sums = demands.sum(dim=1)
    
    # Calculate the heuristic value for each edge based on the following:
    # Promising edges are those that connect a node with a high remaining capacity to a customer with a high demand.
    # We use the normalized demand and the normalized capacity (1 - normalized demand for each node)
    # to determine the heuristic value.
    heuristic_values = (normalized_demands * (1 - row_sums)).unsqueeze(1) * distance_matrix
    
    # Subtract the heuristic values from the distance matrix to get the heuristics tensor
    heuristics = distance_matrix - heuristic_values
    
    # Add a small constant to avoid zeros in the heuristics matrix to ensure numerical stability
    heuristics = heuristics + 1e-6
    
    return heuristics