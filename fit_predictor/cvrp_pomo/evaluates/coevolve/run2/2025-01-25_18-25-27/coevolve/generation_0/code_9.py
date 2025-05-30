import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the sum of distances from the depot (node 0) to all other nodes
    depot_distance_to_all = distance_matrix[0].unsqueeze(0)
    
    # Compute the sum of distances from all other nodes to the depot (node 0)
    all_distance_to_depot = distance_matrix[:, 0].unsqueeze(1)
    
    # Compute the sum of demands of all nodes except the depot
    total_demand = torch.sum(demands[1:])
    
    # Calculate the heuristic values
    heuristics = (depot_distance_to_all + all_distance_to_depot - total_demand) / 2
    
    return heuristics