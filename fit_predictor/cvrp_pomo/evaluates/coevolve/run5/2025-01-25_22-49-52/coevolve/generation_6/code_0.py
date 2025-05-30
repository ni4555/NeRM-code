import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of all demands to normalize them
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the potential value for each edge
    # We use a simple potential value heuristic based on normalized demand
    potential_value = distance_matrix * normalized_demands
    
    # Introduce a small negative value for edges that are not promising
    # This can be adjusted according to the specific problem and desired behavior
    non_promising_penalty = -0.1
    non_promising_edges = (potential_value < 0)
    potential_value[non_promising_edges] += non_promising_penalty
    
    return potential_value