import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that a higher distance value is worse, we use the negative of the distance matrix
    # as the heuristic to reflect that.
    # We use the ratio of the demand to the total capacity as the weight for the heuristic.
    total_capacity = demands.sum()
    demand_weights = demands / total_capacity
    
    # The heuristic is calculated as the negative of the distance multiplied by the demand weight.
    # This means edges with higher distances or higher demand will have lower heuristic values.
    heuristics = -distance_matrix * demand_weights.unsqueeze(1)
    
    return heuristics