import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Create a demand matrix by concatenating the demand vector with itself as the transpose
    demand_matrix = torch.cat((normalized_demands, normalized_demands.unsqueeze(0)), dim=1)
    demand_matrix = demand_matrix.unsqueeze(1).repeat(1, demand_matrix.shape[1], 1)

    # Compute the potential (promising edges will have negative values, undesirable ones positive)
    # This uses the formula (demand_i * demand_j) - distance_ij
    potential = (demand_matrix * demand_matrix.transpose(1, 2)) - distance_matrix

    return potential