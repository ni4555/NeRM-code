import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the demands are normalized
    if not torch.allclose(demands.sum(), torch.tensor(1.0)):
        raise ValueError("Demands must be normalized to a sum of 1.")
    
    # Calculate the total distance for each edge
    total_distance = torch.sum(distance_matrix, dim=0)
    
    # Calculate the sum of demands for each edge
    edge_demand_sum = torch.sum(distance_matrix * demands, dim=0)
    
    # Calculate the demand density for each edge
    demand_density = edge_demand_sum / total_distance
    
    # Node partitioning heuristic
    # Assign a weight based on the ratio of total distance to the sum of demands
    weights = (total_distance / (edge_demand_sum + 1e-10)) * 10  # Adding a small constant to avoid division by zero
    
    # Demand relaxation heuristic
    # Add a penalty for edges with high demand density
    penalty = demand_density * 0.1
    
    # Combine weights and penalties to get the heuristic value for each edge
    heuristic_values = weights - penalty
    
    return heuristic_values