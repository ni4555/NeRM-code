import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the savings for each edge (i, j) where i is not the depot
    savings = distance_matrix[:, 1:] + distance_matrix[1:, :] - 2 * distance_matrix.diagonal()
    
    # Calculate the total demand for each customer (excluding the depot)
    total_demand = demands[1:]
    
    # Calculate the potential savings for each edge by considering the demand
    potential_savings = savings * (1 - total_demand)
    
    # Subtract the demand from the savings to prioritize edges with higher savings
    adjusted_savings = potential_savings - total_demand
    
    # Add a penalty for edges to the depot to discourage unnecessary trips
    adjusted_savings[:, 0] -= 1e5  # Large negative value for edges to the depot
    adjusted_savings[0, :] -= 1e5  # Large negative value for edges from the depot
    
    return adjusted_savings