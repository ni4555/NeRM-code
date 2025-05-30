import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize a tensor with zeros with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic value for each edge
    # Here we use a simple heuristic: the negative of the distance to the depot
    # and the normalized demand multiplied by a scaling factor
    heuristics[:, 0] = -distance_matrix[:, 0]
    heuristics[1:, 0] = heuristics[1:, 0] - 0.1 * normalized_demands[1:]
    
    # Add a penalty for edges that might violate the capacity constraints
    # Here we add a penalty for customer demands that exceed 1 (normalized)
    heuristics[1:, 1:] = heuristics[1:, 1:] - 0.5 * (normalized_demands[1:] > 1)
    
    # Normalize the heuristics to ensure that the values are within a certain range
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)
    
    return heuristics