import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demand vector is of the same shape as the distance matrix
    assert distance_matrix.shape[0] == demands.shape[0]
    
    # Create a cumulative demand mask to evaluate the load distribution along routes
    cumulative_demand_mask = (torch.cumsum(demands, dim=0) <= 1)
    
    # Calculate the load at each node by subtracting the cumulative demand from the capacity
    load_at_node = (demands - cumulative_demand_mask.cumsum(dim=0))
    
    # Create an edge feasibility mask to evaluate the impact of adding an edge on the vehicle's capacity
    edge_feasibility_mask = (distance_matrix < 1) & (load_at_node[:, None] - distance_matrix < 1)
    
    # Initialize the heuristics matrix with high negative values
    heuristics_matrix = -torch.ones_like(distance_matrix)
    
    # Vectorized implementation of the heuristic calculation
    for i in range(edge_feasibility_mask.shape[0]):
        # Get the indices where the edge is feasible and calculate the heuristics value
        feasible_edges = torch.nonzero(edge_feasibility_mask[i], as_tuple=False)
        if feasible_edges.numel() > 0:
            heuristics_matrix[i, feasible_edges[:, 1]] = 1 - load_at_node[i]
    
    return heuristics_matrix