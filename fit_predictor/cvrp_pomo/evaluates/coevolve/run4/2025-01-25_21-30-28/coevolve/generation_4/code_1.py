import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demands vector
    total_demand = demands.sum()
    
    # Normalize the demands vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Create a vector of ones with the same shape as the demands vector
    ones = torch.ones_like(normalized_demands)
    
    # Calculate the sum of each row in the distance matrix
    row_sums = distance_matrix.sum(dim=1)
    
    # Calculate the sum of each column in the distance matrix
    col_sums = distance_matrix.sum(dim=0)
    
    # Calculate the sum of the normalized demands for each edge
    demand_sums = (normalized_demands * ones).sum(dim=0)
    
    # Create a heuristics matrix with negative values for undesirable edges
    heuristics = -distance_matrix
    
    # Modify the heuristics matrix to have positive values for promising edges
    heuristics = heuristics * (1 - (demands > 0).float())
    
    # Adjust the heuristics based on the sum of demands for each edge
    heuristics = heuristics + demand_sums
    
    # Adjust the heuristics based on the row sums to favor shorter paths
    heuristics = heuristics - row_sums
    
    # Adjust the heuristics based on the column sums to favor shorter paths
    heuristics = heuristics - col_sums
    
    return heuristics