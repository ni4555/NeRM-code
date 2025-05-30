import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on the normalized demands and distance matrix
    # Here we use a simple heuristic: the product of the distance and the normalized demand
    # This heuristic assumes that the more distant and higher demand nodes are less desirable
    heuristics = distance_matrix * normalized_demands

    return heuristics