import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix is symmetric (since it's a distance matrix)
    distance_matrix = (distance_matrix + distance_matrix.t()) / 2
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity (assuming total_demand is 1 for simplicity)
    normalized_demands = demands / total_demand
    
    # Compute the heuristics using the formula: heuristics = -distance + demand
    # We want to encourage routes with lower distances and higher demands
    heuristics = -distance_matrix + normalized_demands
    
    # Add a penalty to discourage zero distance edges (which would be invalid in CVRP)
    zero_distance_penalty = torch.min(distance_matrix) * 0.1
    heuristics = torch.clamp(heuristics, min=zero_distance_penalty)
    
    return heuristics