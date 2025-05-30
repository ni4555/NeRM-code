import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along the diagonal
    cumulative_demand = torch.sum(demands, dim=0)
    
    # Calculate the cumulative distance from the depot to each customer
    cumulative_distance = torch.sum(distance_matrix[:, 1:], dim=0)
    
    # Normalize the cumulative demand by the total vehicle capacity
    normalized_demand = cumulative_demand / demands[0]
    
    # Create a mask for the edges where the normalized demand is less than 1 (promising)
    # and a large negative value for edges where the normalized demand is greater than 1 (undesirable)
    edge_mask = -torch.ones_like(distance_matrix)
    edge_mask[1:, 1:] = torch.where(normalized_demand[1:] < 1, -normalized_demand[1:], edge_mask[1:, 1:])
    
    # Add the cumulative distance to the promising edges
    edge_mask[1:, 1:] += cumulative_distance
    
    # Return the heuristics matrix
    return edge_mask