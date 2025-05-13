import random
import pandas as pd
from copy import deepcopy
from library.solution import Solution
from typing import Callable


def get_best_ind(population: list[Solution], maximization: bool):
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

    # 2. Repeat until termination condition
    for gen in range(1, max_gen + 1):
        if verbose:
            print(f"-------------- Generation: {gen} --------------")

        # 2.1. Create an empty population P'
        new_population = []

        # 2.2. If using elitism, insert best individual from P into P'
        if elitism:
            new_population.append(deepcopy(get_best_ind(population, maximization)))

        # 2.3. Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # 2.3.1. Choose 2 individuals from P using a selection algorithm
            first_ind = selection_algorithm(population, maximization)
            second_ind = selection_algorithm(population, maximization)

            if verbose:
                print(f"Selected individuals:\n{first_ind}\n{second_ind}")

            # 2.3.2. Choose an operator between crossover and replication
            # 2.3.3. Apply the operator to generate the offspring
            if random.random() < xo_prob:
                offspring1_repr, offspring2_repr = first_ind.crossover(second_ind)
                if verbose:
                    print(f"Applied crossover")
            else:
                offspring1_repr, offspring2_repr = deepcopy(first_ind), deepcopy(second_ind)
                if verbose:
                    print(f"Applied replication")

            if verbose:
                print(f"Offspring:\n{offspring1_repr}\n{offspring2_repr}")

            # 2.3.4. Apply mutation to the offspring
            first_new_ind = offspring1_repr.mutation(mut_prob)
            # 2.3.5. Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if verbose:
                print(f"First mutated individual: {first_new_ind}")

            if len(new_population) < len(population):
                second_new_ind = offspring2_repr.mutation(mut_prob)
                new_population.append(second_new_ind)
                if verbose:
                    print(f"Second mutated individual: {first_new_ind}")

        # 2.4. Replace P with P'
        population = new_population

        if verbose:
            print(
                f"Final best individual in generation: {get_best_ind(population, maximization).fitness()}"
            )

        # Store best individual for each generation
        generation_best_scores_df.iloc[gen - 1, 0] = gen
        generation_best_scores_df.iloc[gen - 1, 1] = get_best_ind(
            population, maximization
        ).fitness()

    # 3. Return the best individual in P
    return get_best_ind(population, maximization), generation_best_scores_df
