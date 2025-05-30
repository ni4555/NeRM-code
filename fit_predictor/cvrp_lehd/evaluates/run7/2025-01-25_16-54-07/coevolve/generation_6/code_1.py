import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demand for each customer to represent relative contribution to the total load
    demand_normalized = demands / demands.sum()
    
    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over all possible edges, except self-loops
    for i in range(n):
        for j in range(1, n):
            # Calculate the potential increase in cumulative demand when including edge (i, j)
            cumulative_demand_increase = (demand_normalized[i] * distance_matrix[i, j])
            # Add the potential increase to the corresponding heuristic value
            heuristics[i, j] = cumulative_demand_increase
    
    return heuristics