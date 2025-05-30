import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize demands to account for vehicle capacity
    demand_factor = 1.0 / demands
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Promote edges to the depot
    heuristics[:, 0] = -distance_matrix[:, 0]
    heuristics[0, :] = -distance_matrix[0, :]
    
    # Add a small penalty to edges that exceed capacity (demand_factor < 1 indicates excess capacity)
    # The penalty is a large negative value that can be compared to other heuristic values
    heuristics[demand_factor < 1] *= -1000
    
    # Use demand factor to scale heuristics, promoting edges with lower demands
    heuristics *= demand_factor
    
    return heuristics