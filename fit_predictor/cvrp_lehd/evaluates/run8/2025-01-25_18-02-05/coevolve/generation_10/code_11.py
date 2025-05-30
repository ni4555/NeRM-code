import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demands = demands / demands.sum()

    # Calculate the "promise" for each edge by considering the distance and the negative demand (which encourages
    # avoiding high demand edges)
    promise_matrix = -distance_matrix * demands.unsqueeze(1) - demands.unsqueeze(0)

    # Optionally, we can use some simple heuristics to adjust the values further, such as:
    # - Inverse demand: The higher the demand, the more "promise" the edge has to be included
    inverse_demand = 1 / (demands + 1e-8)  # Adding a small constant to avoid division by zero
    promise_matrix += distance_matrix * inverse_demand.unsqueeze(1) * inverse_demand.unsqueeze(0)

    return promise_matrix