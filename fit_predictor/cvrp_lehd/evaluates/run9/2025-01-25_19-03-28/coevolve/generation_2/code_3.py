import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic for each edge
    # We use a simple heuristic based on the ratio of customer demand to distance to the depot
    # This heuristic assigns higher values to edges with lower demand relative to distance
    heuristics = (normalized_demands / distance_matrix) * -1  # Negative values for undesirable edges
    
    return heuristics