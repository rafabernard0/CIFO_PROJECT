import random
from copy import deepcopy
from library.solution import Solution


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
