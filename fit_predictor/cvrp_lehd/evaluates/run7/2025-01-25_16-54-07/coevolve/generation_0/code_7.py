import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the total vehicle capacity is normalized to 1 in the demand vector
    # Compute the cumulative demand
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Create a mask where a value is positive if the cumulative demand at that node
    # is less than the vehicle capacity, and negative otherwise
    mask = cumulative_demand < 1
    
    # Use the mask to create a new distance matrix where we subtract the distance
    # if the edge is promising (cumulative demand is less than capacity), and add
    # a large negative value if it's undesirable (cumulative demand is greater than
    # capacity). The subtraction and addition of a large negative value helps to
    # prioritize edges that are within the capacity constraint.
    heuristics_matrix = torch.where(mask, -distance_matrix, torch.full_like(distance_matrix, -1e6))
    
    return heuristics_matrix