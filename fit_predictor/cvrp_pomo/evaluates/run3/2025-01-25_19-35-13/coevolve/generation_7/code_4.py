import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of normalized demands for each edge
    edge_demand_sum = demands.unsqueeze(1) + demands.unsqueeze(0)
    
    # Normalize edge demands relative to the total vehicle capacity (assuming demands are normalized already)
    # This step assumes the total demand is 1, hence we don't divide by the sum of demands
    edge_demand_normalized = edge_demand_sum / 2
    
    # Incorporate distance and road quality factors (assuming these are given as multipliers)
    distance_factor = torch.exp(-distance_matrix / 100)  # Example: exponential decay of distance
    road_quality_factor = torch.tensor([1 if i == j else 0.5 for i, j in enumerate(distance_matrix)])  # Example: road quality
    road_quality_factor = road_quality_factor.unsqueeze(1) + road_quality_factor.unsqueeze(0)
    
    # Calculate the potential function for each edge
    potential = edge_demand_normalized * (distance_factor * road_quality_factor)
    
    # Introduce a small constant to prevent division by zero errors
    epsilon = 1e-6
    potential = torch.clamp(potential, min=epsilon)
    
    # Convert potential to heuristics by negating undesirable edges (negative potential) and leaving positive edges as is
    heuristics = -torch.log(-potential)
    
    return heuristics