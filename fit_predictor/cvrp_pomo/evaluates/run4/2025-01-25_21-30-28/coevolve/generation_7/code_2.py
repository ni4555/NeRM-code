import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the total demand which is used for normalization
    total_demand = demands.sum().item()
    
    # Calculate the inverse distance heuristic
    # This is the negative distance because we want to prioritize shorter distances
    inv_distance = -distance_matrix
    
    # Calculate the demand normalization heuristic
    # We divide the demands by the total demand and subtract from 1 to normalize them
    # This helps in balancing the allocation of customer demands
    demand_normalization = (1 - (demands / total_demand))
    
    # Combine both heuristics using element-wise addition
    # The resulting heuristic values will be negative for undesirable edges
    combined_heuristics = inv_distance + demand_normalization
    
    return combined_heuristics