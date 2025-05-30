import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    demands = demands.to(distance_matrix.dtype)
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by total capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic for each edge
    # The heuristic is defined as the negative sum of the demands of the two nodes
    # This encourages edges to connect nodes with complementary demands
    heuristics = -torch.abs(normalized_demands[torch.arange(n)] + normalized_demands[torch.arange(1, n)])
    
    # Incorporate distance into the heuristic
    heuristics += distance_matrix
    
    # Adjust the heuristic for the depot node (0) to be less promising
    # This is done by adding a large negative value to the diagonal of the matrix
    large_negative = -1e8
    heuristics[torch.arange(n), torch.arange(n)] += large_negative
    
    return heuristics