# 🎵 Computational Intelligence for Optimization: Music Festival Lineup

This is our project repository for **Computational Intelligence for Optimization**, where we tackled the **Music Festival Lineup** problem using evolutionary algorithms. This repository includes everything from problem description and datasets to our implementation, experimental notebooks, and results.

---

## 📌 Project Overview

The goal of this project is to design an **optimal lineup** for a music festival by assigning **35 artists** to specific time slots and stages. The lineup must:

- Maximize **artist popularity** during prime time slots  
- Ensure **genre diversity** in each time slot  
- Minimize **conflicts** between artists with overlapping audiences  

These objectives often conflict, making the problem a great candidate for **evolutionary algorithms** that explore trade-offs in complex solution spaces.

---

## 🗂 Repository Structure

This repository is organized into the following directories and files:

### 📁 `data/`
Contains the input data for the optimization problem:
- `artists.csv`: List of artists with their **name**, **genre**, and **popularity score** (0–100).
- `conflicts.csv`: **Pairwise conflict matrix** between artists (values from 0 to 1).

### 📁 `library/`
Source code for the optimization library, organized into:
- `GA/`: Genetic algorithm and best-individual functions
- `lineup/`: Problem definition and solution interface
- `mutation_funcs/`: Mutation strategies
- `performance_analysis/`: Evaluation tools for experiment performance
- `selection_algorithms/`: Selection strategies
- `solution/`: Core solution class
- `tuning_funcs/`: Utilities for hyperparameter tuning
- `xo_funcs/`: Crossover strategies

### 📁 `combination_search/`
CSV files containing the results of various experiment configurations. These are used for performance analysis in the notebook.

### 📓 `For Tuning.ipynb`
Notebook to test and tune different algorithm configurations.

### 📓 `Performance.ipynb`
Notebook to evaluate, compare, and visualize the performance of different experimental results.

---

## 🚀 Getting Started

1. Clone the repository.
2. Install any necessary dependencies listed in the notebooks or library modules.
3. Run the `For Tuning.ipynb` notebook to explore algorithm settings.
4. Use the `Performance.ipynb` notebook to evaluate and visualize outcomes.

---

## 📂 Explore & Contribute

Feel free to explore the repository, run the notebooks, and use the code for your own experiments.  
We welcome feedback, suggestions, and contributions!
