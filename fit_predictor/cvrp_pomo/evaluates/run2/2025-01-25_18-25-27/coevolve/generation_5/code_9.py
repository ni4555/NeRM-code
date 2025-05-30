import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to have a sum of 1
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Calculate the potential cost of visiting each customer
    # Subtracting the demand value from the distance to make higher demands more promising
    potential_costs = distance_matrix - normalized_demands.unsqueeze(1)

    # Use a penalty for high demands to avoid overloading the vehicle
    penalty = torch.clamp(potential_costs, min=0)  # Ensure no negative values

    return penalty