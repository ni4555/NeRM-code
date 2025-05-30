import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand sum
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values
    # For this simple heuristic, we use the normalized demand as the heuristic value
    # since it reflects the potential of the edge to contribute to the load of the vehicle.
    heuristics = normalized_demands
    
    return heuristics