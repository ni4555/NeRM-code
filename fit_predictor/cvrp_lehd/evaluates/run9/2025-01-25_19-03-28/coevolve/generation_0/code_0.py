import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demand vector does not include the depot demand (index 0)
    demands = demands[1:]
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per customer
    average_demand = total_demand / len(demands)
    
    # Calculate the difference from average demand for each customer
    demand_diff = demands - average_demand
    
    # Calculate the potential utility of each customer based on demand difference
    # Customers with higher demand difference are potentially more critical
    utility = demand_diff**2
    
    # Normalize the utility by the total utility and subtract to create a heuristic
    # Negative values will indicate less promising edges
    utility_normalized = utility / utility.sum()
    heuristics = -utility_normalized
    
    # Create a mask for the edges from the depot to other customers
    mask = (distance_matrix > 0)
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Assign the heuristics values to the mask locations
    heuristics[mask] = heuristics_v2
    
    return heuristics