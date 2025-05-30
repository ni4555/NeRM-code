import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands tensor is broadcastable to the shape of the distance matrix
    demands = demands.unsqueeze(0).expand_as(distance_matrix)
    
    # Compute the absolute difference in demands
    demand_diff = torch.abs(demands - demands.transpose(0, 1))
    
    # Combine the distance and the demand difference to compute the heuristic
    # You can adjust the weight of the distance and demand difference as needed
    weight_distance = 1.0
    weight_demand = 1.0
    heuristic_matrix = weight_distance * distance_matrix + weight_demand * demand_diff
    
    # Set a negative value for the diagonal to avoid including the depot in the solution
    # This is a common practice in VRP heuristics
    torch.fill_diagonal_(heuristic_matrix, -float('inf'))
    
    return heuristic_matrix