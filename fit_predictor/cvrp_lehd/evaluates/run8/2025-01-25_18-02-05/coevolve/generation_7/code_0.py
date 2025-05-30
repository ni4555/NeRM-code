import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demands = demands / demands.sum()
    
    # Compute the heuristics matrix
    # We will use the following heuristic: (1 / distance) * demand
    # We use a large negative constant for the diagonal to avoid self-assignment
    negative_constant = -1e10
    identity_matrix = torch.eye(distance_matrix.shape[0]).to(distance_matrix.device)
    heuristic_matrix = (1 / (distance_matrix + negative_constant)) * demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # Add a large negative constant to the diagonal to ensure no self-assignment
    heuristic_matrix = heuristic_matrix + identity_matrix * negative_constant
    
    return heuristic_matrix