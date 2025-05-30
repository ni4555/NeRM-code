import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Calculate the cumulative demand at each node considering the distance to the next node
    cumulative_demand = torch.cumsum(distance_matrix * normalized_demands, dim=1)
    
    # Subtract the demand of the current node from the cumulative demand to get the additional demand
    additional_demand = cumulative_demand - demands.unsqueeze(1)
    
    # Create a mask to ensure that additional demand does not exceed vehicle capacity
    capacity_mask = (additional_demand < 1.0).float()  # Assuming vehicle capacity is 1 for normalization
    
    # Calculate the heuristics as a combination of negative distance (to encourage shorter paths) and capacity-based prioritization
    heuristics = -distance_matrix + capacity_mask
    
    return heuristics