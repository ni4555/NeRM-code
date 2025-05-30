import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand from the starting node (index 0)
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative distance from the starting node
    cumulative_distance = torch.cumsum(distance_matrix[:, 0], dim=0)
    
    # Calculate the cumulative cost (demand * distance)
    cumulative_cost = cumulative_demand * cumulative_distance
    
    # Calculate the cumulative cost divided by the total demand to normalize
    normalized_cumulative_cost = cumulative_cost / cumulative_demand
    
    # Calculate the cumulative cost divided by the cumulative distance to normalize
    normalized_cumulative_distance = cumulative_cost / cumulative_distance
    
    # Calculate the sum of the normalized cumulative cost and normalized cumulative distance
    combined_normalized_cost = normalized_cumulative_cost + normalized_cumulative_distance
    
    # Subtract the maximum value from the combined normalized cost to ensure negative values for undesirable edges
    max_combined_normalized_cost = combined_normalized_cost.max()
    combined_normalized_cost = combined_normalized_cost - max_combined_normalized_cost
    
    return combined_normalized_cost