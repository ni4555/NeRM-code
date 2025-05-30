import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each edge
    edge_demand_sums = torch.matmul(normalized_demands.unsqueeze(1), demands.unsqueeze(0)).squeeze(1)
    
    # Normalize the distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Compute the heuristic value as the negative of the sum of edge demand sums and normalized distances
    heuristic_matrix = -edge_demand_sums - normalized_distance_matrix
    
    return heuristic_matrix