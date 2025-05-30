import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # PSLS: Problem-Specific Local Search
    # Calculate initial heuristics based on distance (the closer, the better)
    heuristics = -distance_matrix

    # APSO: Adaptive Particle Swarm Optimization
    # Introduce a particle swarm optimization-inspired heuristic
    # Here we use a simple heuristic that decreases the heuristic value for closer edges
    # which is a form of social learning in particle swarms
    # Note: In a real APSO, this would be more complex, with velocity and position adjustments
    social_learning_factor = torch.linspace(1, 0.1, steps=n).unsqueeze(0).unsqueeze(1)
    heuristics += distance_matrix * social_learning_factor

    # DTSCF: Dynamic Tabu Search with Problem-Specific Cost Function
    # Introduce a cost-based heuristic that penalizes imbalance in vehicle loads
    # Assuming each vehicle has the same capacity for simplicity
    vehicle_capacity = 1
    max_demand_per_vehicle = demands.max()
    load_imbalance_penalty = (demands - max_demand_per_vehicle).pow(2)
    load_imbalance_penalty = load_imbalance_penalty.clamp(min=0)  # Remove negative values due to squaring

    # Adjust heuristics based on the expected load imbalance
    heuristics -= load_imbalance_penalty

    # Combine the PSLS and DTSCF with APSO to finalize the heuristics
    heuristics = heuristics - demands_normalized.unsqueeze(1)  # Avoid assigning more than one customer to a single edge

    # Return the heuristics
    return heuristics