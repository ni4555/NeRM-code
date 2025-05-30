import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer node
    normalized_demands = demands / demands.sum()
    
    # Calculate the sum of distances for each row (from the depot to each customer)
    row_sums = distance_matrix.sum(dim=1)
    
    # Calculate the sum of distances for each column (from each customer to the depot)
    col_sums = distance_matrix.sum(dim=0)
    
    # Create a mask for the diagonal (the distance from the depot to itself)
    diagonal_mask = torch.eye(distance_matrix.shape[0], dtype=torch.bool)
    
    # Calculate the total cost for each edge
    total_costs = row_sums + col_sums
    
    # Calculate the cost to visit each customer (including the return to the depot)
    cost_to_visit = total_costs - row_sums
    
    # Combine the costs with the normalized demands to get the heuristic values
    heuristic_values = cost_to_visit * normalized_demands
    
    # Replace the diagonal elements with negative infinity since we don't want to visit the depot twice
    heuristic_values[diagonal_mask] = -float('inf')
    
    return heuristic_values