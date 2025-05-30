import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    demands = demands / demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the cost of visiting each customer from the depot
    cost_from_depot = distance_matrix[0]
    
    # Calculate the cost of returning to the depot from each customer
    cost_to_depot = distance_matrix[:, 0]
    
    # Compute the heuristic value for each edge
    # The heuristic is the negative of the sum of the costs to visit and return to the depot
    heuristics = -cost_from_depot - cost_to_depot
    
    # Adjust the heuristics for customer demands
    # The heuristic for edges connecting to customers with higher demands should be lower
    heuristics += demands
    
    return heuristics