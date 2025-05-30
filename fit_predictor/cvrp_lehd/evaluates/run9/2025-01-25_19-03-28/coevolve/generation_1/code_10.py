import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance for each edge, which is a negative indicator
    total_distance = -distance_matrix.sum(dim=1, keepdim=True)
    
    # Normalize the customer demands by the total vehicle capacity
    demand_normalized = demands / demands.sum()
    
    # Calculate the potential benefit of each edge (customer demand times distance)
    potential_benefit = demands * distance_matrix
    
    # Normalize the potential benefit by the total vehicle capacity
    normalized_potential_benefit = potential_benefit / demands.sum()
    
    # Combine the negative total distance and normalized potential benefit
    heuristics = total_distance + normalized_potential_benefit
    
    return heuristics