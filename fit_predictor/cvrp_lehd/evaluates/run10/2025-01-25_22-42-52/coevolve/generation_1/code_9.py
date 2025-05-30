import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands to identify the balance of load between nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the total demand in the matrix to normalize the difference
    total_demand = demands.sum()
    
    # Normalize the demand difference by the total demand to get a relative difference
    normalized_demand_diff = demand_diff / total_demand
    
    # Calculate the negative of the distance matrix to make the lower distances more desirable
    negative_distance_matrix = -distance_matrix
    
    # Combine the normalized demand difference with the negative distance matrix
    # This heuristic encourages the selection of edges with a balanced load and short distances
    combined_heuristic = negative_distance_matrix + normalized_demand_diff
    
    # Replace negative values with zeros (undesirable edges)
    combined_heuristic = torch.clamp(combined_heuristic, min=0)
    
    return combined_heuristic