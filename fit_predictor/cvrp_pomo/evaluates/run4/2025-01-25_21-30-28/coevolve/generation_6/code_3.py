import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized inverse distance heuristics
    # Inverse distance is a heuristic where we assume that closer customers are more likely to be visited first.
    # We normalize it to ensure that it doesn't exceed 1 and to account for the capacity constraint.
    inv_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    demand_ratio = demands / demands.sum()  # Normalize demands
    normalized_inv_distance = inv_distance * demand_ratio
    
    # Calculate the normalization heuristic
    # This heuristic scales the inverse distance by the ratio of the demand of the customer to the total demand.
    # This helps in prioritizing customers with higher demands.
    normalization = demands / demands.sum()
    
    # Combine the heuristics
    # We sum the heuristics to get a more balanced heuristic that incorporates both distance and demand.
    combined_heuristics = normalized_inv_distance + normalization
    
    # Convert any negative values to zero (undesirable edges)
    desirable_edges = torch.clamp(combined_heuristics, min=0)
    
    return desirable_edges