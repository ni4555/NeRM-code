import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    inverse_distance = 1 / (distance_matrix ** 2)

    # Incorporate demand into the heuristic
    demand_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Combine the heuristics
    combined_heuristic = inverse_distance + demand_heuristic

    # Enforce capacity constraints with a dynamic demand penalty function
    # For simplicity, we can use a linear penalty function here
    # In a real-world scenario, this would be more complex and adaptive
    load_penalty = torch.clamp(demands.unsqueeze(1) + demands.unsqueeze(0), max=1)
    capacity_penalty = load_penalty * -1000  # Negative values for undesirable edges

    # Final heuristic with penalties
    final_heuristic = combined_heuristic + capacity_penalty

    return final_heuristic