import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demands_normalized = demands / vehicle_capacity

    # Calculate the potential negative impact of each edge
    negative_impact = distance_matrix.clone() * demands_normalized.unsqueeze(1)
    negative_impact = negative_impact + (1 - demands_normalized.unsqueeze(1))

    # Calculate the positive impact of each edge
    positive_impact = torch.exp(-distance_matrix / 10.0)  # Exponential decay with a small scale factor

    # Combine the negative and positive impacts
    heuristics = negative_impact - positive_impact

    # Normalize the heuristics to ensure consistent performance across varying problem scales
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())

    return heuristics