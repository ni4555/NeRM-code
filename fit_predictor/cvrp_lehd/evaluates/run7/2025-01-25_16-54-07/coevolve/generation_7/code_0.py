import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand difference between each pair of nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Compute the cumulative demand masks for each edge
    # This is a heuristic that assumes a high potential for an edge if the cumulative demand
    # along the edge is significantly lower than the maximum demand
    max_demand = demands.max()
    cumulative_demand = torch.cumsum(demand_diff, dim=1)
    cumulative_demand_masks = (cumulative_demand < max_demand.unsqueeze(1)).float()
    
    # Compute the heuristic based on the cumulative demand masks and distance matrix
    # This heuristic assumes that edges with lower cumulative demand and higher distance
    # are less promising, hence we subtract distance to get positive values for promising edges
    heuristics = -cumulative_demand_masks * distance_matrix
    
    # Normalize the heuristics to ensure all values are within a specific range
    # This step is optional and can be adjusted based on the problem context
    max_heuristic = heuristics.max()
    heuristics = heuristics / max_heuristic
    
    return heuristics