import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the product of distance and a function of demand
    # Here, we use a simple function: (1 - demand) * distance
    # This function will decrease the value for edges with high demand
    # and increase the value for edges with low demand.
    # The '1 - demand' term is used to ensure that higher demands
    # result in higher negative values, which are undesirable.
    demand_weight = 1 - demands
    weighted_distance = distance_matrix * demand_weight

    # Return the weighted distance matrix
    return weighted_distance