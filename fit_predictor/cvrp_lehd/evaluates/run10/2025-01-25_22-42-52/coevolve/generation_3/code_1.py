import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands for each edge
    edge_demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the edge demand difference squared (for later use in normalization)
    edge_demand_diff_squared = edge_demand_diff ** 2
    
    # Calculate the sum of the squared differences along the diagonal (self-loops)
    diagonal_sum = torch.sum(edge_demand_diff_squared.diag())
    
    # Normalize the squared differences by the sum of the diagonal
    normalized_demand_diff = edge_demand_diff_squared / diagonal_sum
    
    # Calculate the heuristic based on the distance and normalized demand difference
    heuristics = distance_matrix + normalized_demand_diff
    
    # Ensure that the heuristics have the same shape as the distance matrix
    heuristics = heuristics.view_as(distance_matrix)
    
    return heuristics