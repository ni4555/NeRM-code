import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands and the sum of distances from the depot to all customers
    total_demand = demands.sum()
    distance_to_all_customers = distance_matrix[0, 1:].sum()
    
    # Normalize the distance by the total demand to get a relative measure
    normalized_distance = distance_matrix[0, 1:] / total_demand
    
    # Calculate the difference between the sum of demands and the sum of distances
    # This represents the potential savings of visiting a customer early in the route
    potential_savings = (total_demand - distance_to_all_customers) * normalized_distance
    
    # Calculate the sum of demands for all customers (excluding the depot)
    demand_sum = demands[1:].sum()
    
    # Calculate the heuristics based on the potential savings and the normalized demand
    heuristics = potential_savings / demand_sum
    
    return heuristics