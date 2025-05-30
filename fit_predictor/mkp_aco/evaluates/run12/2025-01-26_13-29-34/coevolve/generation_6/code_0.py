import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    assert weight.shape == (n, m) and weight.sum(axis=1) == np.ones(n), "Weight constraint not satisfied"
    
    # Calculate utility based on inverse weight to prioritize light items
    utility = prize / (weight * 10)  # Adjust the scale factor for better performance
    
    # Integrate neural reinforcement learning for real-time adaptation
    # This is a placeholder for the actual neural network part
    # neural_learning_module = ...  # Define your neural network model here
    # adapted_utility = neural_learning_module.adapt(utility)
    
    # Placeholder for adaptive multi-dimensional constraint validation
    # This is a placeholder for the actual algorithm
    # multi_dim_validation_module = ...  # Define your constraint validation here
    # valid_utility = multi_dim_validation_module.validate(utility)
    
    # Use Particle Swarm Optimization (PSO) to optimize the selection
    # Define the PSO parameters
    num_particles = 30
    num_iterations = 100
    pso = ...  # Define your PSO algorithm here
    
    # Perform PSO
    pso.initialize_particles(utility, n)
    for _ in range(num_iterations):
        pso.update_particles(utility)
    
    # Use the best solution found by PSO as the heuristic
    best_solution = pso.get_best_solution()
    
    return best_solution
