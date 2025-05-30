import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand per unit distance for each edge
    demand_per_unit_distance = demands / distance_matrix
    
    # Calculate the maximum demand that can be covered by one trip (capacity normalized)
    max_demand_per_trip = demands.max()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Set the values in the heuristics matrix for edges that are promising
    # We consider edges promising if their demand per unit distance is less than or equal to the maximum demand per trip
    heuristics[demand_per_unit_distance <= max_demand_per_trip] = 1
    
    return heuristics