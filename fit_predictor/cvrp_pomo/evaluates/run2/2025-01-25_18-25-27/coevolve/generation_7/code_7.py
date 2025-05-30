import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize demands to a common level
    normalized_demands = demands / total_demand
    
    # Calculate the demand heuristic (promising edges will have higher values)
    demand_heuristic = normalized_demands * (distance_matrix ** 2)
    
    # Add a penalty for high demands to discourage them (negative values)
    # Here, we use a simple penalty that is the negative of the demand
    demand_penalty = -normalized_demands
    
    # Combine the demand heuristic and the penalty
    combined_heuristic = demand_heuristic + demand_penalty
    
    return combined_heuristic