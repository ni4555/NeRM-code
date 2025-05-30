import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that the demands are normalized by the total vehicle capacity
    # and the distance matrix is symmetric and the diagonal elements are 0.
    # The heuristic is based on the inverse of the distance (promising edges) and
    # the inverse of the demand (promising edges) to avoid overloading the vehicles.
    # Negative values for undesirable edges are not included as per the problem description.
    
    # Calculate the inverse of the demand vector
    demand_inverse = 1 / (demands - 1e-8)  # Adding a small constant to avoid division by zero
    
    # Calculate the inverse of the distance matrix
    distance_inverse = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Element-wise multiplication of the inverse demands and the inverse distances
    combined_heuristics = demand_inverse * distance_inverse
    
    return combined_heuristics