import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands as the denominator for normalization
    demand_sum = demands.sum()
    
    # Calculate the relative demand for each customer
    relative_demands = demands / demand_sum
    
    # Calculate the heuristic value for each edge
    # Heuristic: demand contribution minus distance (promising edges have positive values)
    # Negative values are used for undesirable edges (e.g., if distance is infinite or too large)
    heuristics = relative_demands * demand_sum - distance_matrix
    
    # Ensure that the values are within the range that can be used effectively
    # Negative values can be set to a very low negative value to represent undesirable edges
    heuristics = torch.clamp(heuristics, min=-1e9)
    
    return heuristics