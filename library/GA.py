import random
import pandas as pd
from copy import deepcopy
from library.solution import Solution
from typing import Callable
from tqdm.auto import tqdm


def get_best_individual(population: list[Solution], maximization: bool):
    fitness_list = [ind.fitness() for ind in population]
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]


def genetic_algorithm(
    initial_population: list[Solution],
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = True,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: bool = True,
    verbose: bool = False,
):
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
    """
    # 1. Initialize a population with N individuals
    population = initial_population

    # Initialize Generation DataFrame
    generation_best_scores_df = pd.DataFrame(
        columns=["Generation", "Fitness"], index=[i for i in range(max_gen)]
    )

    # 2. Repeat reach max_gen generations
    outer_bar = tqdm(
        range(1, max_gen + 1),
        desc="Generations",
        unit="gen",
        position=0,
        leave=True,
    )

    inner_bar = tqdm(
        total=len(initial_population),
        desc="Gen  1",
        unit="ind",
        position=1,
        leave=False,
    )

    for gen in outer_bar:
        inner_bar.reset(total=len(population))
        inner_bar.set_description(f"Gen {gen:2d}")

        # Create an empty population P'
        new_population = []

        if verbose:
            print(f"-------------- Generation: {gen} --------------")

        # 2.2. If using elitism, insert best individual from P into P'
        if elitism:
            new_population.append(
                deepcopy(get_best_individual(population, maximization))
            )
            inner_bar.update()

        # 2.3. Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # 2.3.1. Choose 2 individuals from P using a selection algorithm
            p1 = selection_algorithm(population, maximization)
            p2 = selection_algorithm(population, maximization)

            if verbose:
                tqdm.write(f"Selected:\n{p1.repr}\n{p2.repr}")

            # -- Crossover / replication
            if random.random() < xo_prob:
                child1, child2 = p1.crossover(p2)
                if verbose:
                    tqdm.write("Applied crossover")
            else:
                child1, child2 = deepcopy(p1), deepcopy(p2)
                if verbose:
                    tqdm.write("Applied replication")

            # -- Mutation + insertion
            new_population.append(child1.mutation(mut_prob))
            inner_bar.update()

            if len(new_population) < len(population):
                new_population.append(child2.mutation(mut_prob))
                inner_bar.update()

        population = new_population

        best = get_best_individual(population, maximization)
        generation_best_scores_df.iloc[gen - 1] = (gen, best.fitness())

        if verbose:
            tqdm.write(f"Best fitness in generation {gen}: {best.fitness():.5f}")

    inner_bar.close()  # tidy the second bar
    return get_best_individual(population, maximization), generation_best_scores_df
