import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate cumulative demand mask
    cum_demand = torch.cumsum(demands, dim=0)
    
    # Create edge feasibility mask to evaluate the impact of adding an edge
    edge_feasibility = (distance_matrix < cum_demand) & (distance_matrix < 1)  # Assuming 1 is the maximum capacity
    
    # Calculate the contribution of each edge to the load distribution
    # Here we use the cumulative demand as the priority factor
    priority = cum_demand / distance_matrix
    
    # Combine edge feasibility and priority to get the heuristics
    heuristics = (edge_feasibility * priority). squeeze()
    
    return heuristics