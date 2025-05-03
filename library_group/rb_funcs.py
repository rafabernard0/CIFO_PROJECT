def random_initial_solution(artists_df, 
                            n_stages = 5, 
                            n_slots = 7):
    """
    Generate a random valid initial lineup matrix where each artist is assigned
    to exactly one stage and one time slot.

    Parameters:
    - artists_df: DataFrame containing the list of artists 
    - n_stages: int, 5 stages as default
    - n_slots: int, 7 time slots as default

    Returns:
    - matrix: np.ndarray of shape (n_stages, n_slots) with shuffled artist identifiers
    """
    import pandas as pd
    import numpy as np

    total_artists = artists_df.shape[0]
    total_required = n_stages * n_slots

    # Ensure that the number of artists matches the required number of slots
    assert total_artists == total_required, (
        f"Expected {total_required} artists for a {n_stages}x{n_slots} matrix, "
        f"but got {total_artists} artists."
    )

    # Extract artist identifiers (fallback to first column if 'artist_id' or 'name' not found)
    if 'artist_id' in artists_df.columns:
        artist_ids = artists_df['artist_id'].tolist()
    elif 'name' in artists_df.columns:
        artist_ids = artists_df['name'].tolist()
    else:
        artist_ids = artists_df.iloc[:, 0].tolist()

    # Shuffle the list of artists
    np.random.shuffle(artist_ids)

    # Reshape into a matrix of shape (n_stages, n_slots)
    matrix = np.array(artist_ids).reshape((n_stages, n_slots))

    return matrix


def display_matrix_as_table(matrix, stage_prefix='Stage', slot_prefix='Slot'):
    """
    Display a matrix of artists as a nicely formatted DataFrame
    with labels for stages and time slots (like a spreadsheet).

    Parameters:
    - matrix: np.ndarray of shape (n_stages, n_slots)
    - stage_prefix: str, prefix for row labels (e.g., 'Stage')
    - slot_prefix: str, prefix for column labels (e.g., 'Slot')

    Returns:
    - DataFrame styled as a table
    """
    import pandas as pd

    n_stages, n_slots = matrix.shape

    # Create row and column labels
    row_labels = [f"{stage_prefix} {i+1}" for i in range(n_stages)]
    col_labels = [f"{slot_prefix} {j+1}" for j in range(n_slots)]

    # Create DataFrame
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    
    return df



def two_phase_shuffle_mutation( individual_matrix, 
                                fitness_func, 
                                base_mutation_prob=0.3, 
                                current_fitness=None, 
                                max_fitness=None,
                                attempts_per_phase=30):
    """
    Perform a two-phase shuffle mutation: first by columns, then by rows.
    
    Parameters:
    - individual_matrix: np.ndarray of shape (n_stages, n_slots)
    - fitness_func: callable that returns fitness score
    - base_mutation_prob: float, initial mutation probability
    - current_fitness: float, fitness of the current solution
    - max_fitness: float, best possible fitness (used for adaptive probability)
    - attempts_per_phase: int, number of mutation attempts per phase
    
    Returns:
    - final_matrix: np.ndarray, the mutated matrix
    - final_fitness: float, fitness score of the mutated matrix
    """
    import numpy as np
    import random

    def adaptive_prob(base, current, max_):
        if current is not None and max_ is not None:
            progress = current / max_
            return base * (1 - progress)
        return base

    def shuffle_axis(matrix, axis, prob, attempts):
        best = matrix.copy()
        best_fitness = fitness_func(best)

        for _ in range(attempts):
            if random.random() > prob:
                continue  # skip mutation attempt based on probability

            candidate = best.copy()

            if axis == 'column':
                for col in range(candidate.shape[1]):
                    np.random.shuffle(candidate[:, col])
            elif axis == 'row':
                for row in range(candidate.shape[0]):
                    np.random.shuffle(candidate[row, :])

            if not is_valid(candidate):
                continue

            candidate_fitness = fitness_func(candidate)
            if candidate_fitness > best_fitness:
                best = candidate
                best_fitness = candidate_fitness

        return best, best_fitness

    # Calcular probabilidade adaptativa
    mutation_prob = adaptive_prob(base_mutation_prob, current_fitness, max_fitness)

    # Fase 1: shuffle por colunas
    phase1_matrix, phase1_fitness = shuffle_axis(
        matrix=individual_matrix,
        axis='column',
        prob=mutation_prob,
        attempts=attempts_per_phase
    )

    # Fase 2: shuffle por linhas
    final_matrix, final_fitness = shuffle_axis(
        matrix=phase1_matrix,
        axis='row',
        prob=mutation_prob,
        attempts=attempts_per_phase
    )

    return final_matrix, final_fitness


def is_valid(matrix):
    flat = matrix.flatten()
    return len(set(flat)) == flat.size