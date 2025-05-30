import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate inverse distance heuristic
    # This is a simple inverse distance heuristic that assigns a higher value to closer edges
    # This is a placeholder for the actual IDH logic which would likely be more complex
    # and would take into account the normalized demands and vehicle capacity constraints.
    idh_values = 1 / distance_matrix
    
    # Apply demand normalization to IDH values
    idh_values = idh_values * normalized_demands
    
    # Introduce a penalty function for capacity constraints
    # Assuming a linear penalty proportional to the distance from the capacity threshold
    # The penalty is subtracted from the IDH values for edges that are too close to the capacity
    capacity_threshold = 0.95  # Example threshold, 95% of vehicle capacity
    penalty = (1 - capacity_threshold) / distance_matrix
    penalty[penalty > 0] = 0  # Set penalty to 0 for edges that are within capacity threshold
    
    # Combine IDH values with the penalty
    combined_heuristics = idh_values - penalty
    
    # Cap the values to ensure they are within a reasonable range
    combined_heuristics = torch.clamp(combined_heuristics, min=-1e10, max=1e10)
    
    return combined_heuristics