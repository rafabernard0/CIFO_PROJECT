# Other libraries
import pandas as pd
import itertools

# Genetic Algorithm
from library.GA import genetic_algorithm

# Class GA Solution
from library.lineup import LUGASolution

# Line-up Problem: Datasets
from library.lineup import artists_df, conflicts_df, HERE


def comb_tuning(
    experience_name: str,
    mut_func: dict,
    xo_func: dict,
    selection_algo: list,
    n_runs: int = 30,
    max_gen: int = 100,
    pop_size: int = 50,
):
    """
    Returns a DataFrame with the following columns:
    - 'Combination' - Combination tuple of a mutation, a crossover and a selection algorithm
    - 'Run' - Each Combination is runned a specified amount of times, this column specifies the run
    number in which the results were obtained.
    - 'Generation' - Each Run is composed of a specified amount of generations, this column specifies the
    generation number in which the results were obtained.
    - 'Best Fitness' - This is the best fitness obtained at each generation.

    Purpose:
    With this df we can play around inter-run and intra-run statistics, since it contains
    all the results from the various combinations
    """

    combinations = list(
        itertools.product(list(mut_func.keys()), list(xo_func.keys()), selection_algo)
    )

    combs_results = []
    for comb in combinations:
        # Store the best results of each generation for each run.
        runs_results = []
        for run in range(1, n_runs + 1):
            initial_population = [
                LUGASolution(mutation_function=comb[0], crossover_function=comb[1])
                for i in range(pop_size)
            ]

            _, generation_df = genetic_algorithm(
                initial_population=initial_population,
                max_gen=max_gen,
                selection_algorithm=comb[2],
                maximization=True,
                xo_prob=xo_func[comb[1]],
                mut_prob=mut_func[comb[0]],
                elitism=True,
                verbose=False,
            )

            current_run = pd.DataFrame()
            current_run["Run"] = [
                run for _ in range(1, max_gen + 1)
            ]  # fill column 'Run' with the #run it is in

            # with run=#run, merge the df that contains the columns with #generation and its best fitness
            one_run = pd.concat([current_run, generation_df], axis=1)
            runs_results.append(one_run)

        current_comb_all_runs = pd.concat(
            runs_results
        )  # concat all the different runs along the index axis

        current_comb = pd.DataFrame()
        current_comb["Combination"] = [
            (comb[0].__name__, comb[1].__name__, comb[2].__name__)
            for _ in range(current_comb_all_runs.shape[0])
        ]
        # combination ID (initials of mutation, xo and selection)
        current_comb["Combination ID"] = [
            str(comb[0].__name__[0] + comb[1].__name__[0] + comb[2].__name__[0]).upper()
            for _ in range(current_comb_all_runs.shape[0])
        ]
        one_comb = pd.concat(
            [
                current_comb.reset_index(drop=True),
                current_comb_all_runs.reset_index(drop=True),
            ],
            axis=1,
        )
        combs_results.append(one_comb)

    final_results = pd.concat(combs_results).reset_index(drop=True)
    final_filepath = (
        HERE.parent / "combination_search" / f"final_results_{experience_name}.csv"
    )
    final_results.to_csv(final_filepath, index=False)

    return final_results


def elitism_tuning(
    mut_func,
    mut_prob,
    xo_func,
    xo_prob,
    selection_algo,
    exp_name: str = "elitism_exp",
    n_runs: int = 30,
    max_gen: int = 100,
    pop_size: int = 100,
):

    comb_results = []
    for elitism in [True, False]:
        # Store the best results of each generation for each run.
        runs_results = []
        for run in range(1, n_runs + 1):
            initial_population = [
                LUGASolution(mutation_function=mut_func, crossover_function=xo_func)
                for i in range(pop_size)
            ]

            _, generation_df = genetic_algorithm(
                initial_population=initial_population,
                max_gen=max_gen,
                selection_algorithm=selection_algo,
                maximization=True,
                xo_prob=xo_prob,
                mut_prob=mut_prob,
                elitism=elitism,
                verbose=False,
            )

            current_run = pd.DataFrame()
            current_run["Run"] = [
                run for _ in range(1, max_gen + 1)
            ]  # fill column 'Run' with the #run it is in

            # with run=#run, merge the df that contains the columns with #generation and its best fitness
            one_run = pd.concat([current_run, generation_df], axis=1)
            runs_results.append(one_run)

        current_comb_all_runs = pd.concat(
            runs_results
        )  # concat all the different runs along the index axis

        current_comb = pd.DataFrame()
        current_comb["Elitism"] = [
            elitism for _ in range(current_comb_all_runs.shape[0])
        ]
        one_comb = pd.concat(
            [
                current_comb.reset_index(drop=True),
                current_comb_all_runs.reset_index(drop=True),
            ],
            axis=1,
        )
        comb_results.append(one_comb)

    final_results = pd.concat(comb_results).reset_index(drop=True)
    final_filepath = HERE.parent / "combination_search" / f"{exp_name}.csv"
    final_results.to_csv(final_filepath, index=False)

    return final_results
