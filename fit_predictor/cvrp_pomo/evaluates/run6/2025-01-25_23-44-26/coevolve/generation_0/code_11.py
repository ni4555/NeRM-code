import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized distance matrix by dividing the original matrix by the max distance
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)
    
    # Calculate the total demand for each node
    total_demand = torch.sum(demands, dim=0)
    
    # Calculate the total demand for each edge (sum of demands of the two nodes connected by the edge)
    edge_demand = demands.unsqueeze(1) + demands.unsqueeze(0)
    
    # Calculate the edge demand to vehicle capacity ratio
    demand_to_capacity_ratio = edge_demand / demands
    
    # Calculate the heuristic value for each edge based on distance and demand to capacity ratio
    heuristics = -normalized_distance_matrix + demand_to_capacity_ratio
    
    # Ensure that the heuristic values are within the specified range (negative for undesirable edges, positive for promising ones)
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)
    
    return heuristics