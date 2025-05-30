import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for the depot
    normalized_demand = demands[0] / demands.sum()
    
    # Calculate the difference in demand between each customer and the normalized demand
    demand_diff = demands - normalized_demand
    
    # Calculate the negative difference as the heuristic for each edge
    heuristics = -torch.abs(demand_diff)
    
    # Normalize the heuristics by the maximum value to ensure all values are positive
    heuristics /= heuristics.max()
    
    # Return the heuristics matrix
    return heuristics