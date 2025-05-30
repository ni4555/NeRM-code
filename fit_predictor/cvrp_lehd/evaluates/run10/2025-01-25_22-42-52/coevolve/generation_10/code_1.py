import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands tensor does not include the depot's demand
    demands = demands[1:]
    
    # Calculate the heuristic values
    distance_penalty = -distance_matrix
    demand_difference = torch.abs(demands - demands[:, None])
    demand_diff_penalty = demand_difference / demands.sum()
    
    # Combine the penalties
    heuristics_values = distance_penalty + demand_diff_penalty
    
    # Adjust the diagonal to be 0, since there's no distance or demand difference from a node to itself
    heuristics_values[torch.eye(len(heuristics_values), dtype=torch.bool)] = 0
    
    return heuristics_values