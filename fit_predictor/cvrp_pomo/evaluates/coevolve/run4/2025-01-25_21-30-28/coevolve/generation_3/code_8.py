import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of demands for each edge
    edge_demands = torch.sum(distance_matrix * demands.unsqueeze(1), dim=0)
    
    # Calculate the heuristic values based on edge weight and demand
    # Here we use a simple heuristic: the more the demand, the more promising the edge
    # We can adjust the weight of demand and distance as needed
    demand_weight = 0.5
    distance_weight = 0.5
    
    heuristics = demand_weight * edge_demands - distance_weight * distance_matrix
    
    # Normalize heuristics to ensure load balancing
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    
    return heuristics