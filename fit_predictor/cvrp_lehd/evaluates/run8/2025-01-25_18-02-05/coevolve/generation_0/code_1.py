import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands to the range [0, 1] based on the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Create a matrix of all ones
    ones_matrix = torch.ones_like(distance_matrix)
    
    # Calculate the cost of not visiting a customer, which is the demand of the customer
    non_visit_cost = normalized_demands * demands
    
    # Calculate the cost of visiting a customer, which is the distance from the depot
    visit_cost = distance_matrix
    
    # Combine the non-visit and visit costs into a single matrix
    combined_costs = non_visit_cost - visit_cost
    
    # Add a small positive constant to avoid division by zero
    epsilon = 1e-8
    combined_costs = combined_costs / (combined_costs.abs() + epsilon)
    
    # Apply a threshold to the combined costs to create the heuristics matrix
    heuristics_matrix = torch.clamp(combined_costs, min=epsilon)
    
    return heuristics_matrix