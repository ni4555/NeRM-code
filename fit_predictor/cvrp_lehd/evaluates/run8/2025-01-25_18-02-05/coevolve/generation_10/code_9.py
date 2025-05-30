import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics based on the distance matrix and normalized demands
    # We are using the following heuristic: a simple function of distance and demand
    # For example, a common heuristic is 1/distance + demand, but this is just a placeholder
    # You can modify the heuristic function as needed for your specific problem
    heuristics = 1.0 / distance_matrix + normalized_demands
    
    # Replace zeros in the distance matrix with a large negative value to mark undesirable edges
    undesirable_edges_mask = distance_matrix == 0
    heuristics[undesirable_edges_mask] = -torch.inf
    
    return heuristics