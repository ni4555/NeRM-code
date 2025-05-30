import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands from the depot to each node
    cumulative_d demands.cumsum(dim=0)
    
    # Calculate the cumulative sum of demands from each node to the depot
    cumulative_d_reverse = demands.cumsum(dim=0)[::-1]
    
    # Calculate the difference between cumulative sums for each edge
    edge_diff = cumulative_d - cumulative_d_reverse
    
    # Normalize the differences by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_diff = edge_diff / total_capacity
    
    # Use a simple heuristic: edges with higher normalized differences are more promising
    # This can be adjusted with different weightings or more complex heuristics
    heuristic_values = normalized_diff * 1000  # Example scaling factor
    
    return heuristic_values