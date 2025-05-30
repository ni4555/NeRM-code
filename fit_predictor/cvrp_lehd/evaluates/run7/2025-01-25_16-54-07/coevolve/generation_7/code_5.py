import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize demands by the sum of demands to get a cumulative demand mask
    cumulative_demand_mask = demands / demands.sum()
    
    # Calculate the load on each edge based on cumulative demand
    load_on_edges = torch.matmul(cumulative_demand_mask, distance_matrix)
    
    # Calculate the load distribution on each vehicle by summing up the load on edges
    # connected to each vehicle (excluding the depot)
    load_distribution = load_on_edges[1:].sum(dim=0)
    
    # Normalize the load distribution to get a load distribution mask
    load_distribution_mask = load_distribution / load_distribution.sum()
    
    # Calculate the heuristic value for each edge
    heuristic_values = (1 / distance_matrix) * load_distribution_mask
    
    # Add negative values for the edges leading from the depot to undesirable edges
    negative_mask = (distance_matrix == 0) | (cumulative_demand_mask < 0.1)  # Arbitrary threshold for undesirable edges
    heuristic_values = heuristic_values * ~negative_mask - (heuristic_values * negative_mask)
    
    return heuristic_values