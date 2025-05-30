import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to the range [0, 1] by dividing by the sum of demands
    normalized_demands = demands / demands.sum()
    
    # Create a matrix where the element at (i, j) is the normalized demand of node j from node i
    # The diagonal elements are set to 0 as they represent the demand of the node itself
    demand_matrix = torch.zeros_like(distance_matrix)
    demand_matrix.fill_diagonal_(0)
    demand_matrix = demand_matrix * normalized_demands
    
    # Calculate the cumulative sum of demands along the rows (from the depot to each customer)
    cumulative_demand = demand_matrix.sum(dim=1)
    
    # Calculate the cumulative sum of demands along the columns (from each customer to the depot)
    cumulative_demand_transposed = demand_matrix.sum(dim=0)
    
    # Calculate the minimum cumulative demand along each path (from the depot to a customer and back)
    min_cumulative_demand = torch.min(cumulative_demand, cumulative_demand_transposed)
    
    # The heuristic value for each edge is the difference between the total distance and the minimum
    # cumulative demand. We subtract this from the total distance to get negative values for
    # undesirable edges and positive values for promising ones.
    heuristics = distance_matrix - min_cumulative_demand
    
    return heuristics