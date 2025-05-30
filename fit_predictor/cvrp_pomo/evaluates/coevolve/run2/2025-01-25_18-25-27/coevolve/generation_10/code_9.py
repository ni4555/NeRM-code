import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the number of customers (excluding the depot)
    num_customers = distance_matrix.size(0) - 1
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Create a vector of all ones for the number of possible routes
    ones = torch.ones(num_customers, dtype=torch.float32)
    
    # Initialize the heuristics matrix with negative values
    heuristics = -torch.ones_like(distance_matrix)
    
    # For each customer, calculate the heuristic based on distance and demand
    for i in range(num_customers):
        # Calculate the difference in normalized demand
        demand_diff = normalized_demands - normalized_demands[i]
        
        # Calculate the heuristic as a function of distance and demand difference
        heuristics[:, i] = distance_matrix[:, i] + demand_diff
    
    return heuristics