import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the cumulative demand along each edge
    cumulative_demand = torch.cumsum(normalized_demands, dim=0)
    
    # Calculate the heuristic values as the negative of the cumulative demand
    # This heuristic encourages routes with lower cumulative demand
    heuristics = -cumulative_demand
    
    # Add small positive values to undesirable edges to avoid them in the search
    # This can be adjusted based on the specific problem's requirements
    undesirable_threshold = 0.5
    heuristics[heuristics > undesirable_threshold] += 1e-6
    
    return heuristics