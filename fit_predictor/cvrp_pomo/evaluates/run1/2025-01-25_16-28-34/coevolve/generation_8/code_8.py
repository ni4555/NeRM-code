import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the potential heuristics
    # For simplicity, we use the following heuristic:
    # The heuristic for an edge (i, j) is the negative of the distance times the demand
    # of customer j at node i, plus a bonus for the depot (i == 0)
    heuristics = -distance_matrix * normalized_demands
    
    # Add a bonus for the depot to encourage visiting it
    heuristics[0, 1:] += 1  # Add bonus for all edges from depot to customers
    heuristics[1:, 0] += 1  # Add bonus for all edges from customers to depot
    
    return heuristics