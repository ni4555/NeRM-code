import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristic_scores = np.zeros(n)
    
    # Initialize with a random subset
    selected_items = set(random.sample(range(n), n // 2))
    
    # Genetic Algorithm Parameters
    population_size = 10
    mutation_rate = 0.05
    generations = 100
    
    # Stochastic Local Search Parameters
    local_search_iterations = 20
    
    # Genetic Algorithm
    for _ in range(generations):
        population = [selected_items for _ in range(population_size)]
        for _ in range(local_search_iterations):
            for individual in population:
                # Mutation
                if random.random() < mutation_rate:
                    item_to_change = random.choice(list(individual))
                    new_item = random.randint(0, n - 1)
                    individual.remove(item_to_change)
                    individual.add(new_item)
                
                # Evaluate the individual
                value = sum(prize[i] for i in individual)
                weight_sum = sum(weight[i, 0] for i in individual)
                heuristic_scores[individual] = value - weight_sum
        
        # Select the best individual
        selected_items = max(population, key=lambda x: heuristic_scores[x])
    
    # Stochastic Local Search
    best_value = sum(prize[i] for i in selected_items)
    best_weight_sum = sum(weight[i, 0] for i in selected_items)
    for _ in range(local_search_iterations):
        for i in selected_items:
            if random.random() < 0.5:
                # Remove item
                selected_items.remove(i)
                weight_sum = sum(weight[j, 0] for j in selected_items)
                if weight_sum + weight[i, 0] <= 1:
                    selected_items.add(i)
            else:
                # Add item
                for j in range(n):
                    if j not in selected_items and weight[j, 0] <= 1 - sum(weight[k, 0] for k in selected_items):
                        selected_items.add(j)
                        weight_sum = sum(weight[k, 0] for k in selected_items)
                        if weight_sum > 1:
                            selected_items.remove(j)
                        break
        
        # Evaluate the new solution
        value = sum(prize[i] for i in selected_items)
        weight_sum = sum(weight[i, 0] for i in selected_items)
        if value > best_value:
            best_value = value
            best_weight_sum = weight_sum
    
    # Calculate heuristic scores based on the best solution found
    for i in range(n):
        if weight[i, 0] <= 1 - best_weight_sum:
            heuristic_scores[i] = prize[i] - weight[i, 0]
    
    return heuristic_scores
