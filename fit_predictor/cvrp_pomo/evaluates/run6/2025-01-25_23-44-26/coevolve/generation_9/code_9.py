import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    # The inverse distance is a simple heuristic that encourages routes to use
    # edges with shorter distances.
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Incorporate demand sensitivity into the heuristic
    # A demand-sensitive penalty mechanism is applied to edges leading to
    # customers with high demand, which discourages overloading vehicles.
    demand_penalty = -demands * (1 / (1 + demands))  # Demand penalty function
    
    # Combine the inverse distance and demand penalty to form the final heuristic
    combined_heuristic = inverse_distance + demand_penalty
    
    return combined_heuristic