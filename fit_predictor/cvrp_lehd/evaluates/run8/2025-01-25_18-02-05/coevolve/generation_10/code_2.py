import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands[0]  # Assuming demands[0] represents the vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the difference between demands and 1 (normalized demand for depot)
    demand_diff = 1 - normalized_demands
    
    # Calculate the heuristics as the sum of the inverse of the distance and the difference in demand
    heuristics = 1 / distance_matrix + demand_diff
    
    # Replace negative values with zero to make all edges promising
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics