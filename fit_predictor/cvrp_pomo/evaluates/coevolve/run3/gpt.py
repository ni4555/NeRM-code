import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_demand = demands.sum()
    normalized_demands = demands / total_demand
    
    # Define weights for demand and distance heuristics
    demand_weight = 0.8
    distance_weight = 0.2
    
    # Apply dynamic programming to find a lower bound on the cost
    dp = torch.zeros((len(demands), len(demands)))
    for d in range(1, len(demands)):
        for s in range(d):
            dp[d, s] = dp[s, d-1] + distance_matrix[s, d]
    
    # Calculate the potential based on the lower bound and demand
    epsilon = 1e-8
    lower_bound = torch.min(dp[:, 0], dp[0, :])
    edge_potential = (lower_bound / (distance_matrix + epsilon)) * torch.pow(normalized_demands, demand_weight)
    edge_potential = edge_potential * (1 - distance_weight) + (1 / (distance_matrix + epsilon)) * torch.pow(normalized_demands, 0.5) * distance_weight
    
    # Add penalties for nodes with high demand and for long distances
    edge_potential = edge_potential - (edge_potential * 0.1 * (demands > 1.5).float())
    edge_potential = edge_potential + (edge_potential * 0.05 * (distance_matrix < 10).float())
    
    return edge_potential
