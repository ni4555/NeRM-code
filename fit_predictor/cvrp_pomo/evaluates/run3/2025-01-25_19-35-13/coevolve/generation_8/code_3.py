import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands[0]  # Assuming the first node is the depot and has no demand, so demand vector starts from the second node
    total_demand = demands[1:].sum()
    normalized_demands = demands[1:] / total_demand

    # Calculate the heuristics based on normalized demands and distance
    heuristics = -distance_matrix * normalized_demands.unsqueeze(1)
    
    return heuristics