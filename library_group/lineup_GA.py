import numpy as np
import pandas as pd

class LUGASolution(LUSolution):
    def __init__(self,
                 mutation_function, # calable
                 crossover_function, # calable
                 repr: np.ndarray = None,
                 artists_df: pd.DataFrame = artists_df,
                 conflicts_df: pd.DataFrame = conflicts_df,
                 stages: int = 5,
                 slots: int = 7,
                ):
        
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function

        super().__init__(repr=repr, artists_df=artists_df, conflicts_df=conflicts_df, stages=stages, slots=slots)

    def mutation(self, mut_prob):
        new_repr = self.mutation_function(self.repr, mut_prob)
        return LUGASolution(distance_matrix=self.distance_matrix, starting_idx=self.starting_idx,
                             repr=new_repr,
                             mutation_function=self.mutation_function, 
                             crossover_function=self.crossover_function
                             )
    def crossover(self, other_solution):
        offspring1_repr, offspring2_repr = self.crossover_function(self.repr, other_solution.repr)

        return (LUGASolution(repr = offspring1_repr,
                             artists_df = self.artists_df,
                             conflicts_df = self.conflicts_df,
                             stages = self.stages,
                             slots = self.slots,
                             mutation_function = self.mutation_function, # calable
                             crossover_function = self.crossover_function, 
                            ),
                LUGASolution(repr = offspring2_repr,
                             artists_df = self.artists_df,
                             conflicts_df = self.conflicts_df,
                             stages = self.stages,
                             slots = self.slots,
                             mutation_function = self.mutation_function, # calable
                             crossover_function = self.crossover_function, 
                            ),
                )