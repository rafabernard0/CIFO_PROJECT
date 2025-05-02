A modular optimization library designed to solve various optimization problems using optimization algorithms.

## How it works

The library follows a structured approach:

1. **Abstract Base Class (Solution)**
- Defines a generic solution structure for all optimization problems.
- Requires implementation of `fitness()` and `random_initial_representation()`.
- Defined in `solution.py`

2. **Problem-Specific Solution Classes**

- Extend `Solution` and implement problem-specific methods.
- Implements the `fitness()` and `random_initial_representation()` methods
- Example: `TSPSolution` for the Traveling Salesperson Problem in `problems/tsp.py`

3. **Problem-Algorithm-Specific Solution Classes**
- Extend a problem solution class and implements methods needed for an optimization algorithm to work
- Example: `TSPHillClimbingSolution` inherits from `TSPSolution` and implements the `get_neighbors()` method. Available in `problems/tsp.py`.

Implementations of the optimization algorithms are also available in the `algorithms` directory.

## Adding a New Optimization Problem

To introduce a new optimization problem:

1. Create a **problem-specific solution class** that extends `Solution`.
2. This class should implement the `fitness()` and `random_initial_representation()` methods specific to the problem.

## Applying an Optimization Algorithm to a New Problem

To optimize a problem using an algorithm:

1. Define a **problem-algorithm-specific solution class** that extends the **problem-specific class**.
2. Implement the required methods for the chosen optimization algorithm.
3. Instantiate an initial solution using this class and pass it as an argument to the optimization algorithm function.






