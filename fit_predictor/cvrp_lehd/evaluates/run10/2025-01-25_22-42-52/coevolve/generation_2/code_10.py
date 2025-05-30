import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the sum of distances for each edge
    edge_distances = distance_matrix.sum(dim=1)
    
    # Normalize the distances by the total distance
    normalized_distances = edge_distances / edge_distances.sum()
    
    # Compute the heuristics by multiplying the normalized demands and distances
    heuristics = normalized_demands.unsqueeze(1) * normalized_distances.unsqueeze(0)
    
    # The heuristic values should be positive for promising edges and negative for undesirable ones
    # This can be achieved by subtracting the sum of all heuristics from each edge's heuristic
    max_heuristic = heuristics.max()
    heuristics = heuristics - max_heuristic
    
    return heuristics