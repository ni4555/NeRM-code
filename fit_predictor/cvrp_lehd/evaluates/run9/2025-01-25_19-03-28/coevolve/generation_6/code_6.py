import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to sum to 1
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Calculate the heuristic values based on normalized demands and distance
    # Promising edges have positive values, undesirable edges have negative values
    # Here, we use a simple heuristic: the inverse of the demand times the distance
    # This encourages selection of edges with lower demand and shorter distance
    heuristics = (1 / (normalized_demands * distance_matrix)).clamp(min=0)

    return heuristics