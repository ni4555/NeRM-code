import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by the total demand
    normalized_demands = demands / total_demand
    
    # Normalize distances by dividing by the maximum distance in the matrix
    max_distance = distance_matrix.max()
    normalized_distances = distance_matrix / max_distance
    
    # Calculate the heuristic values
    # We use the formula: h(e) = demand of node * normalized distance to node
    # Negative values for undesirable edges, positive for promising ones
    heuristics = -normalized_demands * normalized_distances
    
    return heuristics