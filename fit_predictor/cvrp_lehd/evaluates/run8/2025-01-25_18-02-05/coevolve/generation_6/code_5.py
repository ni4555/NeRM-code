import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the distance matrix is a PyTorch tensor and demands are also PyTorch tensors.
    # We calculate the difference in demands to determine which edges could potentially
    # be more beneficial based on the difference in customer demand between two nodes.
    # Negative values for undesirable edges, positive values for promising ones.
    # This assumes that higher differences in demands indicate a potentially higher
    # contribution to the total demand, hence more promising.
    # Normalize by the maximum absolute demand difference to keep the values within a reasonable range.
    
    # Calculate the difference in demands, excluding the depot itself
    demand_diff = demands[1:] - demands[:-1]
    
    # Compute the normalized demand difference
    max_demand_diff = torch.max(torch.abs(demand_diff))
    normalized_demand_diff = demand_diff / max_demand_diff
    
    # Use the negative normalized demand difference as heuristic
    # since we want negative values for undesirable edges
    heuristics = -normalized_demand_diff
    
    return heuristics