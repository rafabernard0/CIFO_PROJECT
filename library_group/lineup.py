"""Problem definition

Description:
Search space:
Representation:
Fitness function:
Goal: Maximize f(x).
"""

from library.solution import Solution
import numpy as np
import pandas as pd
from tabulate import tabulate
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 2. Point at the data folder (one level up, then into data/)
DATA_DIR = HERE.parent / "data"

# 3. Build full paths to each CSV:
artists_csv = DATA_DIR / "artists.csv"
conflicts_csv = DATA_DIR / "conflicts.csv"

# 4. Load them exactly the same, no matter where you call the script from:
artists_df = pd.read_csv(artists_csv, sep=";", index_col=0)
conflicts_df = pd.read_csv(conflicts_csv, sep=";", index_col=0)
conflicts_df = conflicts_df.apply(
    lambda col: (
        col.str.replace(",", ".").astype(float) if col.name != "conflict" else col
    )
)


class LUSolution(Solution):
    def __init__(
        self,
        repr: np.ndarray = None,
        artists_df: pd.DataFrame = artists_df,
        conflicts_df: pd.DataFrame = conflicts_df,
        stages: int = 5,
        slots: int = 7,
    ):

        # --- Input Validation ---
        if not isinstance(stages, int) or stages <= 0:
            raise ValueError("ERROR: stages must be a positive integer")
        self.stages = stages

        if not isinstance(slots, int) or slots <= 0:
            raise ValueError("ERROR: slots must be a positive integer")
        self.slots = slots

        if repr is None:

            if not isinstance(artists_df, pd.DataFrame):
                raise ValueError("ERROR: artists_df must be a pandas DataFrame")

            if (
                "popularity" not in artists_df.columns
                or "genre" not in artists_df.columns
            ):
                raise ValueError(
                    "ERROR: artists_df must contain 'popularity' and 'genre' columns"
                )

            if not pd.api.types.is_integer_dtype(artists_df.index):
                print(
                    "WARNING: artists_df index is not integer type. Ensure it matches artist IDs used in repr."
                )

            if len(artists_df) != len(conflicts_df):
                raise ValueError(
                    "ERROR: artists_df and conflict_df must have same length."
                )

            self.n_artists = artists_df.shape[0]
            self.artists_df = artists_df
            self.conflicts_df = conflicts_df

            self.repr = self.random_initial_representation()

        if repr is not None:
            self.repr = self._validate_repr(repr)

            self.n_artists = self.repr.size

            # Instanciate dataframes to reference values of popularity, genre, artist names
            # but only after repr is defined so random solution is not created
            self.artists_df = artists_df
            self.conflicts_df = conflicts_df

        # Ensure that exist one artist per time slot and stage
        total_required = self.stages * self.slots
        assert self.n_artists == total_required, (
            f"ERROR: Expected {total_required} artists for a {self.stages}x{self.slots} matrix, "
            f"but got {self.n_artists} artists."
        )

    def random_initial_representation(self):
        """
        Generate a random initial representation of the solution.
        """

        # Extract artist identifiers
        artists_ids = [n for n in range(self.n_artists)]

        # Shuffle the list of artists
        np.random.shuffle(artists_ids)

        # Reshape into a matrix of shape (n_stages, n_slots)
        matrix = np.array(artists_ids).reshape((self.stages, self.slots))

        return matrix

    def _validate_repr(self, repr_input):

        if isinstance(repr_input, list):
            try:
                # Attempt conversion, check for ragged lists implicitly
                repr_array = np.array(repr_input)
                # DEBUG PRINT
                print("INFO: Converting your lists of list into a np.array.")
                if repr_array.ndim == 1 and isinstance(
                    repr_array[0], list
                ):  # Check if it became array of objects
                    raise ValueError(
                        "Input list seems ragged (inner lists have different lengths)."
                    )
            except ValueError as e:
                raise ValueError(
                    f"Could not convert list of lists to array. Original error: {e}"
                )
        elif isinstance(repr_input, np.ndarray):
            repr_array = repr_input
        else:
            raise ValueError(
                "Representation must be a 2D numpy array or a list of lists."
            )

        # Check for dimension
        if repr_array.ndim != 2:
            raise ValueError(
                f"Representation must be 2D, but got {repr_array.ndim} dimensions."
            )

        # Check if data type is numeric (integer or float that can be cast)
        if not np.issubdtype(repr_array.dtype, np.number):
            raise ValueError(
                f"Representation elements must be numeric, but got dtype {repr_array.dtype}."
            )

        # Ensure elements are integers (or cast if safe)
        if not np.issubdtype(repr_array.dtype, np.integer):
            if np.all(repr_array == repr_array.astype(int)):
                print("INFO: Casting representation elements to integers.")
                repr_array = repr_array.astype(int)
            else:
                raise ValueError(
                    "Representation contains non-integer numeric values that cannot be safely cast."
                )

        print("DEBUG: Your repr has been validated")
        return repr_array

    def score_prime_slots_popularity(self):
        """
            Prime   Slot   Popularity  :  The  most  popular  artists  should  be  scheduled  in  the  prime
        slots  (the  last  time  slot  on  each  stage).  This  score  is  calculated  by  normalizing  the
        total  popularity  of  artists  in  prime  slots  against  the  maximum  possible  total  popularity
        (e.g. if only most popular artists - score 100 - were scheduled on the prime slot)
        """
        # Extract the popularity of artists in prime slots
        artist_id_in_prime_slot = self.repr[:, -1]

        popularities_in_prime_slot = self.artists_df.loc[
            artist_id_in_prime_slot, "popularity"
        ]

        score = popularities_in_prime_slot.sum() / (
            self.stages * self.artists_df["popularity"].max()
        )

        return round(score, 5)

    def score_genre_diversity(self):
        """
            Genre   Diversity  :  A  diverse  distribution  of  genres  among  stages  in  each  time  slot
        improves   the   festival   experience.   This   score   is   determined   by   normalizing   the
        number  of  unique  genres  in  each  slot  relative  to  the  maximum  possible  number  of
        unique  genres  (e.g.  if  only  different  genres  were  scheduled  in  that  time  slot)  .  Then
        you take the average among time slots.
        """

        total_unique_genres = len(self.artists_df["genre"].unique())

        # Loop trough stages and count unique genres
        unique_genres_per_stage = []

        for stage_comp in self.repr:
            unique_genres = artists_df.loc[stage_comp, "genre"].unique()
            unique_genres_per_stage.append(len(unique_genres))

        # Tranform into np.array to easily normalize
        unique_genres_per_stage = np.array(unique_genres_per_stage)

        # Normalize
        unique_genres_list_normalized = unique_genres_per_stage / total_unique_genres

        score = unique_genres_list_normalized.mean()

        return round(score, 5)

    def penalty_conflict(self):
        """
            Conflict   Penalty  :   Fan   conflicts   occur   when   artists   with   overlapping   audiences
        perform  at  the  same  time  on  different  stages.  This  score  is  calculated  by  normalizing
        the  total  conflict  value  in  each  slot  against  the  worst  possible  conflict  scenario  (e.g.
        where  all  artists  with  maximum  conflict  are  scheduled  together).  Then  you  take  the
        average   among   time   slots.   Since   conflicts  negatively  impact  the  lineup,  this  is  a
        penalization  score.
        """
        artists_per_slot = self.repr.T

        # Normal case 5 + 4 + 3 +2 + 1 = 15
        worst_conflict_per_slot = sum(range(self.stages + 1))

        normalized_slot_conflicts = []
        # Access slot composition
        for artists in artists_per_slot:

            # Select the conflict sub-matrix for artists in this slot using .loc
            sub_matrix = self.conflicts_df.iloc[artists, artists]

            # Sum the upper triangle (excluding diagonal) for total slot conflict
            slot_total_conflict = sub_matrix.values[
                np.triu_indices_from(sub_matrix.values, k=1)
            ].sum()

            # Normalize and save
            normalized_slot_conflicts.append(
                slot_total_conflict / worst_conflict_per_slot
            )

        # Calcualte average
        average_penalty = np.mean(normalized_slot_conflicts)

        return round(average_penalty, 5)

    def fitness(self):
        prime_slot = self.score_prime_slots_popularity()
        genre = self.score_genre_diversity()
        conflict = self.penalty_conflict()

        score = prime_slot + genre - conflict

        return score

    def get_artist_display_map(self):
        """Helper to get the mapping from ID to display string (Name or ID)."""

        artist_display_map = {}
        name_col = None
        if "name" in self.artists_df.columns:
            name_col = "name"
        elif "artist_name" in self.artists_df.columns:
            name_col = "artist_name"

        ids_in_schedule = np.unique(self.repr.flatten())

        if name_col and self.artists_df.index.is_unique:
            try:
                artist_display_map = {
                    id_val: (
                        self.artists_df.loc[id_val, name_col]
                        if id_val in self.artists_df.index
                        else str(id_val)
                    )
                    for id_val in ids_in_schedule
                }
                return artist_display_map
            except Exception:
                pass

        return {id_val: str(id_val) for id_val in ids_in_schedule}

    def __repr__(self):
        """Provides a string representation of the solution schedule and its scores."""
        if (
            not hasattr(self, "repr")
            or not isinstance(self.repr, np.ndarray)
            or self.repr.ndim != 2
        ):
            return f"<LUSolution object (repr not initialized or invalid)>"

        schedule = self.repr
        stages, slots = schedule.shape

        # --- Part 1: Schedule Table ---
        display_map = self.get_artist_display_map()
        table_data = []
        for i in range(stages):
            row_data = [
                display_map.get(schedule[i, j], str(schedule[i, j]))
                for j in range(slots)
            ]
            table_data.append(row_data)

        headers = [f"Slot {j+1}" for j in range(slots)]
        row_indices = [f"Stage {i+1}" for i in range(stages)]

        # I
        table_string = tabulate(
            table_data,
            headers=headers,
            showindex=row_indices,
            tablefmt="fancy_grid",
            stralign="center",
            numalign="center",
        )

        # --- Part 2: Scores ---
        try:
            prime_score = self.score_prime_slots_popularity()
            genre_score = self.score_genre_diversity()
            conflict_penalty = self.penalty_conflict()
            fitness_score = self.fitness()

            scores_info = [
                ("Prime Slot Popularity", prime_score),
                ("Genre Diversity", genre_score),
                ("Conflict Penalty", conflict_penalty),
                ("Total Fitness", fitness_score),
            ]
            max_label_len = 0
            if scores_info:
                max_label_len = max(len(label) for label, _ in scores_info)

            scores_lines = ["\nScores:"]
            for label, value in scores_info:
                scores_lines.append(f"  {label.ljust(max_label_len)} : {value:.5f}")
            scores_string = "\n".join(scores_lines)

        except Exception as e:
            scores_string = f"\nScores: Error calculating scores - {e}"

        # --- Combine all parts ---
        header_info = f"<LUSolution ({stages} Stages, {slots} Slots)>"
        return f"{header_info}\n{table_string}{scores_string}"


class LUGASolution(LUSolution):
    def __init__(
        self,
        mutation_function,  # calable
        crossover_function,  # calable
        repr: np.ndarray = None,
        artists_df: pd.DataFrame = artists_df,
        conflicts_df: pd.DataFrame = conflicts_df,
        stages: int = 5,
        slots: int = 7,
    ):

        self.mutation_function = mutation_function
        self.crossover_function = crossover_function

        super().__init__(
            repr=repr,
            artists_df=artists_df,
            conflicts_df=conflicts_df,
            stages=stages,
            slots=slots,
        )

    def mutation(self, mut_prob):
        new_repr = self.mutation_function(self.repr, mut_prob)
        return LUGASolution(
            mutation_function=self.mutation_function,  # calable
            crossover_function=self.crossover_function,  # calable
            repr=new_repr,
            artists_df=self.artists_df,
            conflicts_df=self.conflicts_df,
            stages=self.stages,
            slots=self.slots,
        )

    def crossover(self, other_solution):
        offspring1_repr, offspring2_repr = self.crossover_function(
            self.repr, other_solution.repr
        )

        return (
            LUGASolution(
                repr=offspring1_repr,
                artists_df=self.artists_df,
                conflicts_df=self.conflicts_df,
                stages=self.stages,
                slots=self.slots,
                mutation_function=self.mutation_function,  # calable
                crossover_function=self.crossover_function,
            ),
            LUGASolution(
                repr=offspring2_repr,
                artists_df=self.artists_df,
                conflicts_df=self.conflicts_df,
                stages=self.stages,
                slots=self.slots,
                mutation_function=self.mutation_function,  # calable
                crossover_function=self.crossover_function,
            ),
        )
