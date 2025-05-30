import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load on each vehicle if we were to visit all customers
    total_load = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_load
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the cost of visiting each customer
    # This is a simple heuristic that assumes the cost is equal to the distance
    heuristics += distance_matrix
    
    # Adjust the heuristics based on the normalized demands
    # Customers with higher normalized demand get a lower heuristic value
    heuristics -= normalized_demands
    
    # Apply a penalty for long distances
    # This encourages the algorithm to find more compact routes
    heuristics += torch.log(distance_matrix + 1e-10)  # Adding a small epsilon to avoid log(0)
    
    return heuristics