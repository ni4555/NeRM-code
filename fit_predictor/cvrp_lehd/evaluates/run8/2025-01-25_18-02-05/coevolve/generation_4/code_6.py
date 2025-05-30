import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    demand_sum = demands.sum()
    
    # Normalize the demands to get the demand density for each customer
    demand_density = demands / demand_sum
    
    # Calculate the heuristics based on the demand density
    # The heuristic for an edge is the negative of the demand density of the customer at the destination node
    heuristics = -demand_density[distance_matrix.nonzero(as_tuple=True)[1]]
    
    # Expand the heuristics to match the shape of the distance matrix
    heuristics = heuristics.view(distance_matrix.shape)
    
    # Fill in the diagonal (edges from a node to itself) with a large negative value to make them undesirable
    torch.fill_diagonal_(heuristics, float('-inf'))
    
    return heuristics