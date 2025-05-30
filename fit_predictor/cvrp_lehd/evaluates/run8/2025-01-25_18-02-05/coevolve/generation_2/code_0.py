import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that the demands are normalized by the total vehicle capacity
    # and that the depot node (index 0) is not to be included in the heuristics
    n = distance_matrix.shape[0]
    max_demand = demands.max()
    
    # Calculate the potential heuristic value for each edge (i, j)
    # by considering the distance and the negative demand (promising edges have higher negative values)
    heuristic_matrix = -distance_matrix + demands.unsqueeze(1) * demands.unsqueeze(0)
    
    # To ensure that the depot (node 0) does not get included in the solution,
    # we can set the heuristic values to a very low negative value for the edges leading to the depot
    # and the edges originating from the depot.
    # We do not need to set the diagonal elements to any specific value as they represent the distance from a node to itself.
    for i in range(n):
        if i == 0:
            # Set all incoming edges to the depot to a very low negative value
            heuristic_matrix[0, i] = -max_demand
        if i == n - 1:
            # Set all outgoing edges from the depot to a very low negative value
            heuristic_matrix[i, 0] = -max_demand
    
    return heuristic_matrix