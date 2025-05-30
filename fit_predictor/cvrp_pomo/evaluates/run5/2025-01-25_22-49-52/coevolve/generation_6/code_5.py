import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential value for each edge
    # The potential value is a function of the distance and the demand
    # Here, we use a simple heuristic: potential_value = distance - demand
    # This is a basic approach and can be replaced with more complex ones
    potential_value = distance_matrix - normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)
    
    # To ensure that we do not have zero values in the potential matrix (which would make it non-informative),
    # we add a small constant to all values
    epsilon = 1e-8
    potential_value = torch.clamp(potential_value, min=epsilon)
    
    return potential_value