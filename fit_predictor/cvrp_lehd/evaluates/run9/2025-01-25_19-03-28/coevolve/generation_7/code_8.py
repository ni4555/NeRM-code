import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix and demands are both tensors and the demands are normalized
    distance_matrix = distance_matrix.clone().detach().to(torch.float32)
    demands = demands.clone().detach().to(torch.float32)
    
    # Calculate the demand contribution to the heuristics (using normalized demands)
    demand_contrib = 1.0 / (demands + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Calculate the heuristics based on distance and demand contributions
    # For each edge, calculate the heuristics value as the difference between the distance
    # and a weighted demand contribution
    heuristics = distance_matrix - demand_contrib.unsqueeze(1) * demand_contrib.unsqueeze(0)
    
    return heuristics