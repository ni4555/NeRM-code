import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    demand_sum = demands.sum()
    
    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / demand_sum
    
    # Calculate the potential cost of visiting a node based on its demand and distance
    potential_costs = distance_matrix * normalized_demands
    
    # Adjust the potential costs by subtracting the maximum demand from each node's demand
    adjusted_costs = potential_costs - demands
    
    # Apply a threshold to promote or demote edges based on their adjusted cost
    threshold = demands.max()  # Assuming we want to promote edges with costs lower than the maximum demand
    heuristics = torch.where(adjusted_costs < threshold, adjusted_costs, -adjusted_costs)
    
    return heuristics