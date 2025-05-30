import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the distance matrix is symmetric
    # Calculate the difference in demands between each customer pair
    demand_diff = demands[:, None] - demands[None, :]
    
    # Calculate the absolute difference
    abs_demand_diff = torch.abs(demand_diff)
    
    # Calculate the maximum demand in the row (from depot to each customer)
    max_demand_row = torch.max(abs_demand_diff, dim=1)[0]
    
    # Calculate the maximum demand in the column (from each customer to depot)
    max_demand_col = torch.max(abs_demand_diff, dim=0)[0]
    
    # Calculate the minimum of the max demands in both directions
    min_max_demand = torch.min(max_demand_row, max_demand_col)
    
    # Calculate the heuristics based on the distance and demand difference
    heuristics = distance_matrix - min_max_demand
    
    # Normalize the heuristics to have a range of values
    min_heuristic = heuristics.min()
    max_heuristic = heuristics.max()
    heuristics = (heuristics - min_heuristic) / (max_heuristic - min_heuristic)
    
    return heuristics