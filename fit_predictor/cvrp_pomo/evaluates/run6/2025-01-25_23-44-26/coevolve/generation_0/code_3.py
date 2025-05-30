import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Calculate the negative of the normalized demand for each edge as a heuristic
    # This will encourage the heuristic to favor edges with lower demand
    edge_heuristics = -normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Add a small constant to prevent division by zero
    epsilon = 1e-6
    edge_heuristics = edge_heuristics + epsilon
    
    # Calculate the distance between each pair of nodes and add to the heuristic
    edge_heuristics = edge_heuristics + distance_matrix
    
    # Ensure that the diagonal elements (self-loops) are set to a very negative value
    # to discourage them from being included in the solution
    diagonal_mask = torch.eye(edge_heuristics.shape[0], dtype=edge_heuristics.dtype)
    edge_heuristics = edge_heuristics - 2 * diagonal_mask
    
    return edge_heuristics