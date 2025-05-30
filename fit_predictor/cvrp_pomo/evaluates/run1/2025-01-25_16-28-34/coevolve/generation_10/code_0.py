import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands from the end to the start
    # This will help us to determine the potential load at each customer
    cumulative_demands = torch.cumsum(demands[::-1], dim=0)[::-1]
    
    # Calculate the potential load at each customer
    load_at_customers = cumulative_demands - demands
    
    # Calculate the potential load at each edge (from depot to customer and customer to customer)
    # We will use a negative value for the edge from the depot to the first customer
    edge_loads = torch.cat((load_at_customers, load_at_customers[:-1]), dim=0)
    
    # Calculate the heuristic value for each edge
    # The heuristic is based on the absolute load at the destination customer
    # We use a negative value for the edge from the depot to the first customer
    heuristics = -torch.abs(edge_loads)
    
    # Normalize the heuristics by the maximum absolute value to ensure they are on the same scale
    max_abs_value = torch.max(torch.abs(heuristics))
    if max_abs_value != 0:
        heuristics /= max_abs_value
    
    return heuristics