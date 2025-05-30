import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand for all customers
    total_demand = demands.sum()
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_demand
    
    # Calculate the potential contribution of each edge
    # For this simple heuristic, we'll use the normalized demand of the customers
    # connected by each edge and subtract the distance to penalize longer paths
    edge_potential = (normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
                      - distance_matrix)
    
    # We can add an additional heuristic, for example, a penalty for edges close to the
    # depot since these might require the vehicle to return sooner.
    # Here we're using a simple quadratic function to penalize such edges.
    # The exact form of the penalty can be adjusted based on the specific problem.
    depot_penalty = distance_matrix ** 2
    
    # Combine the heuristics to get the final heuristic values
    heuristic_values = edge_potential - depot_penalty
    
    return heuristic_values