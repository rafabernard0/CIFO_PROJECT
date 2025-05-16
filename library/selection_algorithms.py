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


def linear_ranking_selection(
    population: list,
    maximization: bool,
    s: float = 1.7,
    verbose: bool = False,
):
    """
    Linear ranking selection. Each individual is ranked and given a selection
    probability based on position, not raw fitness.
    """
    if not (1.0 <= s <= 2.0):
        raise ValueError("s must be between 1.0 and 2.0")

    N = len(population)

    # Sort population by fitness (descending if maximization, ascending otherwise)
    sorted_pop = sorted(
        population,
        key=lambda ind: ind.fitness(),
        reverse=maximization,
    )

    # Assign linear ranking probabilities
    probabilities = [
        ((2 - s) / N) + (2 * i * (s - 1)) / (N * (N - 1)) for i in range(N)
    ]

    if verbose:
        for i, (ind, prob) in enumerate(zip(sorted_pop, probabilities)):
            print(f"Rank {i+1}: Fitness={ind.fitness():.4f}, Prob={prob:.4f}")

    # Select one individual based on probabilities
    selected = random.choices(sorted_pop, weights=probabilities, k=1)[0]

    if verbose:
        print(f"Selected individual: {selected} with fitness {selected.fitness():.4f}")

    return deepcopy(selected)

