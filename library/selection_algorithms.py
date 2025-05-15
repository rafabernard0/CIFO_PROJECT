# Put here selection algorithms functions (Elaborate 2)

import random
from copy import deepcopy

from library.solution import Solution


def fitness_proportionate_selection(population: list[Solution], maximization: bool):
    if maximization:
        fitness_values = []
        for ind in population:
            if ind.fitness() < 0:
                # If fitness is negative (invalid solution like in Knapsack)
                # Set fitness to very small positive value
                # Probability of selecting this individual is nearly 0.
                fitness_values.append(0.0000001)
            else:
                fitness_values.append(ind.fitness())
    else:
        # Minimization: Use the inverse of the fitness value
        # Lower fitness should have higher probability of being selected
        fitness_values = [1 / ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)
    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_fitness)
    # For each individual, check if random number is inside the individual's "box"
    box_boundary = 0
    for ind_idx, ind in enumerate(population):
        box_boundary += fitness_values[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)


def tournament_selection(
    population: list[Solution],
    maximization: bool,
    tournament_size: int = 3,
    verbose: bool = False,
):

    # Check if tournament size is valid
    if tournament_size < 1:
        raise ValueError("tournament_size must be ≥ 1")

    # Select with replacement a random subset of individuals from the population
    tournament = random.choices(population, k=tournament_size)

    if verbose:
        print(
            f"Tournament individuals: {[(ind.repr, ind.fitness())for ind in tournament]}"
        )

    key = lambda ind: ind.fitness()
    winner = max(tournament, key=key) if maximization else min(tournament, key=key)

    if verbose:
        print(
            f"Best individual in tournament: {winner} with fitness {winner.fitness()}"
        )

    return deepcopy(winner)




def ranking_selection(population, fitnesses, s=1.7):
    """
    Perform linear ranking selection.
    Assigns selection probability based on sorted rank, not raw fitness.
    """
    N = len(population)

    # Sort population by fitness (higher is better)
    sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    sorted_population = [x[0] for x in sorted_pop]

    # Assign probabilities using linear ranking formula
    probabilities = [
        ((2 - s) / N) + (2 * i * (s - 1)) / (N * (N - 1)) for i in range(N)
    ]

    # Select one individual based on these probabilities
    selected = random.choices(sorted_population, weights=probabilities, k=1)[0]
    return selected
