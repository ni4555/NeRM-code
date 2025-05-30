import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each edge
    edge_demands = torch.sum(normalized_demands[:, None] * normalized_demands, dim=0)
    
    # Calculate the distance matrix of edge demands
    edge_demand_matrix = torch.cdist(edge_demands, edge_demands)
    
    # Subtract the edge demand matrix from the distance matrix to get the heuristics
    heuristics = distance_matrix - edge_demand_matrix
    
    return heuristics