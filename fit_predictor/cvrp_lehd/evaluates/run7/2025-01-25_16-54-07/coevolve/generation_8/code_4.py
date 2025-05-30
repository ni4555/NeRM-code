import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative load for each edge
    cumulative_load = cumulative_demand.unsqueeze(1) + distance_matrix.unsqueeze(0)
    
    # Calculate the capacity left for each edge
    capacity_left = vehicle_capacity - cumulative_demand.unsqueeze(1)
    
    # Create a mask for edge feasibility
    edge_feasibility_mask = (capacity_left > 0) & (distance_matrix != 0)
    
    # Calculate the contribution of each edge to balanced load distribution
    edge_contribution = (capacity_left * distance_matrix) * edge_feasibility_mask
    
    # Normalize the edge contribution by the maximum possible contribution to get a heuristic value
    max_contribution = edge_contribution.max()
    heuristic_values = edge_contribution / max_contribution
    
    # Invert the sign to have negative values for undesirable edges
    heuristic_values = -1 * heuristic_values
    
    return heuristic_values