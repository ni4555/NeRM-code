import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of distances from the depot to all other nodes (excluding the depot itself)
    depot_to_all = distance_matrix[0, 1:]
    
    # Calculate the sum of demands from the depot to all other nodes (excluding the depot itself)
    demand_to_all = demands[1:]
    
    # Calculate the sum of distances from all nodes to the depot (excluding the depot itself)
    all_to_depot = distance_matrix[1:, 0]
    
    # Calculate the sum of demands from all nodes to the depot (excluding the depot itself)
    all_demand_to_depot = demands[1:]
    
    # Calculate the normalized demands (divided by the total capacity)
    normalized_demands = demands / demands.sum()
    
    # Compute the heuristic values using the following formula:
    # heuristic = - (distance_to_depot + distance_from_depot) * demand_normalized
    heuristic_values = - (depot_to_all + all_to_depot) * normalized_demands
    
    # Compute the heuristic values for the depot itself as 0 (it's not included in the solution)
    heuristic_values = torch.cat([torch.zeros(1), heuristic_values])
    
    # Return the computed heuristic values
    return heuristic_values