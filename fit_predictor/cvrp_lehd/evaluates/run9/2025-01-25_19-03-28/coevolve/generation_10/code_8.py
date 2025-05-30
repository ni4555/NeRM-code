import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    vehicle_capacity = 1.0
    normalized_demands = demands / vehicle_capacity
    
    # Compute the heuristic values using a combination of methods
    # Here, we are using a simple heuristic approach for demonstration:
    # The heuristic value is the negative of the distance (penalizing longer paths)
    # plus a demand-based factor (rewarding lower demand)
    # This is a simple example and can be replaced with more sophisticated heuristics.
    heuristic_values = -distance_matrix + normalized_demands
    
    # Return the heuristic matrix
    return heuristic_values