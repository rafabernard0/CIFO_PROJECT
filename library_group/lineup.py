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

artists_df = pd.read_csv("../data/artists.csv", sep=";", index_col=0)
conflicts_df = pd.read_csv("../ddata/conflicts.csv", sep=";", index_col=0)
conflicts_df = conflicts_df.apply(
    lambda x: x.str.replace(",", ".").astype(float) if x.name != "conflict" else x
)


class LUSolution(Solution):
    def __init__(
        self,
        artists_df: pd.DataFrame = artists_df,
        conflicts_df: pd.DataFrame = conflicts_df,
        repr: str = None,
        stages: int = 5,
        slots: int = 7,
    ):
        self.artists_df = artists_df
        self.conflicts_df = conflicts_df
        self.stages = stages
        self.slots = slots
        self.n_artists = artists_df.shape[0]

        if not isinstance(artists_df, pd.DataFrame):
            raise ValueError("artists_df must be a pandas DataFrame")
        if not isinstance(conflicts_df, pd.DataFrame):
            raise ValueError("conflicts_df must be a pandas DataFrame")
        if not isinstance(stages, int) or stages <= 0:
            raise ValueError("stages must be a positive integer")
        if not isinstance(slots, int) or slots <= 0:
            raise ValueError("slots must be a positive integer")

        # Ensure that exist one artist per time slot and stage
        total_required = self.stages * self.slots
        assert self.n_artists == total_required, (
            f"Expected {total_required} artists for a {self.stages}x{self.slots} matrix, "
            f"but got {self.n_artists} artists."
        )

        if repr:
            repr = self._validate_repr(repr)

        super().__init__(repr=repr)

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

    def _validate_repr(self, repr: str): ...

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
        ].tolist()

        score = sum(popularities_in_prime_slot) / (
            self.stages * max(self.artists_df["popularity"])
        )

        # Maximum possible in that scenario ?

        return score

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

        return score

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

        return average_penalty

    def fitness(self):
        prime_slot = self.score_prime_slots_popularity()
        genre = self.score_genre_diversity()
        conflict = self.penalty_conflict()

        score = prime_slot + genre - conflict

        return score
