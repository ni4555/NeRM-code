import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize heuristic values with the ratio of prize to weight
    heuristic_values = prize / weight
    
    # Apply adaptive constraint-driven filtering by considering only items that meet weight constraints
    filtered_heuristics = heuristic_values * (weight < 1).astype(float)
    
    # Incorporate deep reinforcement learning for real-time decision-making
    # Assuming a pre-trained RL model to get the importance of items
    rl_importance = np.random.rand(len(filtered_heuristics))  # Placeholder for RL model output
    
    # Combine PSO evolutionary swarm intelligence
    # Assuming a PSO algorithm that has been applied to the problem
    pso_values = np.random.rand(len(filtered_heuristics))  # Placeholder for PSO algorithm output
    
    # Combine the heuristic values using a weighted sum approach
    combined_heuristics = filtered_heuristics * rl_importance * pso_values
    
    # Normalize the combined heuristics to get final heuristic values
    max_combined = np.max(combined_heuristics)
    final_heuristics = combined_heuristics / max_combined
    
    return final_heuristics
