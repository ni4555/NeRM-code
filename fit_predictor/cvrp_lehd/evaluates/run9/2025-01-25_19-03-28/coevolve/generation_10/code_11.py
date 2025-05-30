import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the travel distance heuristic
    travel_distance_heuristic = -distance_matrix

    # Calculate the load balance heuristic
    load_balance_heuristic = (demands - demands.mean()) ** 2

    # Calculate the service time heuristic
    service_time_heuristic = (demands / demands.mean()) ** 2

    # Combine heuristics using a weighted sum approach
    # The weights can be adjusted based on the problem's importance
    weights = torch.tensor([0.5, 0.3, 0.2], dtype=travel_distance_heuristic.dtype)
    combined_heuristic = weights[0] * travel_distance_heuristic + \
                          weights[1] * load_balance_heuristic + \
                          weights[2] * service_time_heuristic

    return combined_heuristic