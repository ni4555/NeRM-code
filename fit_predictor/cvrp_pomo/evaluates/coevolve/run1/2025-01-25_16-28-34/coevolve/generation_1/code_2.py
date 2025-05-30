import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between each pair of nodes
    demand_diff = demands[:, None] - demands[None, :]
    
    # The heuristic value for each edge is based on the absolute difference in demands
    # and the distance between the nodes.
    # We use a negative heuristic for edges with high absolute demand difference
    # to discourage them from being included in the solution.
    heuristics = -torch.abs(demand_diff) * distance_matrix
    
    # We could further refine the heuristics by adding a constant based on the total
    # vehicle capacity to ensure that high-demand edges are penalized more.
    # For this example, we'll just return the basic heuristic values.
    
    return heuristics