import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the demands are normalized by the total vehicle capacity
    # and the depot node is indexed by 0, the heuristics can be calculated
    # as the difference between the demand at the destination node and
    # the average demand of all nodes (which should be 0 if demands are normalized)
    
    # Calculate the average demand
    average_demand = demands.mean()
    
    # Generate the heuristics based on the difference in demands
    heuristics = demands - average_demand
    
    # For the depot node (index 0), set a fixed value since it is the starting point
    heuristics[0] = 0
    
    return heuristics