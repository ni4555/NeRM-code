import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are on the same device
    demands = demands.to(distance_matrix.device)
    
    # The cost matrix initialized with a large negative value for undesirable edges
    cost_matrix = torch.full_like(distance_matrix, fill_value=-float('inf'))
    
    # The cost for an edge (i, j) is the sum of the distance and the demands of the two nodes
    cost_matrix = distance_matrix + demands + demands[:, None]
    
    # We use a small value to represent the capacity of the vehicle (for comparison)
    vehicle_capacity = demands.sum()
    
    # Find the edges where the total demand does not exceed the vehicle capacity
    valid_edges = (cost_matrix < vehicle_capacity)
    
    # We can also introduce a small penalty for edges that go to the depot to prioritize leaving the depot
    depot_penalty = torch.full_like(distance_matrix, fill_value=-0.1)
    cost_matrix = cost_matrix + depot_penalty
    
    # Set the cost for valid edges to 0
    cost_matrix[valid_edges] = 0
    
    return cost_matrix