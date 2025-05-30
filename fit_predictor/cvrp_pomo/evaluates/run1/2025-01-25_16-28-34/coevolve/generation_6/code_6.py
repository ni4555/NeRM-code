import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Problem-specific Local Search: Calculate the total demand to each customer
    total_demand = torch.sum(demands)
    demand_potential = demands / total_demand
    
    # Adaptive PSO Population Management: Use a simple heuristic based on proximity to the depot
    # (Here we use a simple inverse distance heuristic, this could be replaced with more complex ones)
    distance_to_depot = torch.abs(distance_matrix[:, 0])
    distance_potential = 1 / (distance_to_depot + 1e-8)  # Add a small constant to avoid division by zero
    
    # Dynamic Tabu Search with Adaptive Cost Function: Introduce a cost that penalizes high load
    # Here we use a simple heuristic that considers the difference between max and min demand
    demand_difference = torch.max(demands) - torch.min(demands)
    load_potential = -demand_difference  # Negative value to prefer lower load
    
    # Combine the heuristics using weights (weights can be adjusted based on problem specifics)
    alpha, beta, gamma = 0.5, 0.3, 0.2  # Example weights
    combined_potential = alpha * demand_potential + beta * distance_potential + gamma * load_potential
    
    return combined_potential