import plotly.graph_objects as go
import pandas as pd
from scipy.stats import mannwhitneyu

from typing import List, Tuple, Dict, Any
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ABF_MBF(
    df, metric, std=False, exp_comb=None, comb_col=None, plot_by_exp=False
):
    """
    Plots average/median best fitness per generation for each configuration.

    Parameters:
    - df: Dataset containing fitness data
    - metric: receives either 'ABF' (Average Best Fitness) or 'MBF' (Median Best Fitness) as arguments
    - std: If True it plots conjointly in the plot
    - exp_comb: To set the specific experience/combination in the title.
    - comb_col: Column Name that we want to group by. Only set when wanting to group by another column
    or even that the column name is not the default. If given, exp_comb should not be given and the title
    will remain simple.

        Specific Parameters to change only when calling this function inside the comb_between_exp function:
            - plot_by_exp: Change to True to plot based on different experiments
    """

    # when the comb_col is not specified (the usual)
    if not comb_col:
        if plot_by_exp:
            comb_col = "Experience"
            title_col = "Combination"
        else:
            comb_col = "Combination"
            title_col = "Experience"

    # define if it will group by the average or median
    if metric == "ABF":
        xbf_df = df.groupby([comb_col, "Generation"])["Fitness"].mean().reset_index()
        title = (
            f"<b>Average Best Fitness per Generation</b><br>{title_col} {exp_comb}"
            if exp_comb
            else "<b>Average Best Fitness per Generation</b>"
        )
    if metric == "MBF":
        xbf_df = df.groupby([comb_col, "Generation"])["Fitness"].median().reset_index()
        title = (
            f"<b>Median Best Fitness per Generation</b><br>{title_col} {exp_comb}"
            if exp_comb
            else "<b>Median Best Fitness per Generation</b>"
        )

    if std:
        # it is equal either plotting mbf or abf
        std_df = df.groupby([comb_col, "Generation"])["Fitness"].std().reset_index()
        std_df.rename(columns={"Fitness": "std"}, inplace=True)
        std_col = std_df[["std"]].copy()
        xbf_df = pd.concat(
            [xbf_df, std_col], axis=1
        )  # adding the column for the fitness std

        xbf_df["y_upper"] = xbf_df["Fitness"] + xbf_df["std"]
        xbf_df["y_lower"] = xbf_df["Fitness"] - xbf_df["std"]
        xbf_df.loc[xbf_df["y_lower"] < 0, "y_lower"] = 0

    combs = df[comb_col].unique()

    fig = go.Figure()

    for comb in combs:
        data = xbf_df[xbf_df[comb_col] == comb]
        y_vals = data["Fitness"].values

        fig.add_trace(
            go.Scatter(
                x=xbf_df["Generation"].values,
                y=y_vals,
                mode="lines+markers",
                name=str(comb),
                showlegend=True,
                hovertemplate=f"{comb_col}: {comb}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>",
            )
        )
        if std:
            fig.add_trace(
                go.Scatter(
                    x=xbf_df["Generation"].values,
                    y=data["y_upper"],
                    mode="lines",
                    name="+1 std Train",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xbf_df["Generation"],
                    y=data["y_lower"],
                    mode="lines",
                    name="-1 std Train",
                    fill="tonexty",
                    fillcolor="rgba(0,0,255,0.1)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        title_font=dict(family="Arial", size=18),
        xaxis=dict(
            title="Generation",
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Fitness",
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        legend=dict(
            title=dict(text=f"{comb_col}", side="top", font=dict(size=12)),
            orientation="h",
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        autosize=True,
        height=650,
        width=850,
    )

    fig.show()


def plot_BF_inter_run(df, comb_ids: list = None, exp_comb: str = None):
    """
    Plots best fitness per run for each configuration.
    The objective is to evaluate the stability of the combination by seeing if
    it maintains the results consistenly.

    Parameters:
    - df: Dataset containing fitness data
    """

    bf_run = df.groupby(["Combination", "Run"])["Fitness"].max().reset_index()
    if comb_ids:
        comb_id_pair = df[["Combination", "Combination ID"]].drop_duplicates()
        comb_id_pair_filtered = comb_id_pair[
            comb_id_pair["Combination ID"].isin(comb_ids)
        ]
        combs = comb_id_pair_filtered["Combination"].unique()
    else:
        combs = df["Combination"].unique()

    fig = go.Figure()

    for comb in combs:
        data = bf_run[bf_run["Combination"] == comb]
        y_vals = data["Fitness"].values

        fig.add_trace(
            go.Scatter(
                x=bf_run["Run"].values,
                y=y_vals,
                mode="lines+markers",
                name=str(comb),
                hovertemplate=f"Combination: {comb}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=(
            dict(
                text=f"<b>Best Fitness per Run</b><br>Experience:{exp_comb}",
                x=0.5,
                xanchor="center",
            )
            if exp_comb
            else dict(text="<b>Best Fitness per Run</b>", x=0.5, xanchor="center")
        ),
        xaxis=dict(
            title="Run",
            gridcolor="lightgray",
            tickmode="linear",  # Ensures ticks are shown at regular intervals
            dtick=1,
            range=[0, None],
        ),
        yaxis=dict(
            title="Best Fitness",
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        legend=dict(
            title=dict(
                text="Combinations",  # Added bold for emphasis
                side="top",  # Positions title above the legend items
                font=dict(size=12),
            ),
            orientation="h",
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        autosize=True,
        height=650,
        width=850,
    )

    fig.show()


def plot_BF_inter_run_boxplot(df, comb_ids: list = None, exp_comb: str = None):
    """
    Plots boxplots of best fitness per run for each configuration.
    The objective is to evaluate the stability and distribution of results across runs.

    Parameters:
    - df: Dataset containing fitness data
    - comb_ids: List of combination IDs to filter by (optional)
    - exp_comb: Experiment combination name for title (optional)
    """

    # get the best fitness per run for each combination
    bf_run = (
        df.groupby(["Combination", "Combination ID", "Run"])["Fitness"]
        .max()
        .reset_index()
    )

    # filter combinations if IDs are provided
    if comb_ids:
        bf_run = bf_run[bf_run["Combination ID"].isin(comb_ids)]

    fig = go.Figure()

    # get unique combinations
    combs = bf_run["Combination ID"].unique()

    for comb in combs:
        data = bf_run[bf_run["Combination ID"] == comb]
        comb_str = data[data["Combination ID"] == comb][["Combination"]].iloc[0, 0]

        fig.add_trace(
            go.Box(
                y=data["Fitness"],
                name=f"{comb}: {comb_str}",
                x=[comb] * len(data),
                marker=dict(size=5),
                line=dict(width=1),
                hoverinfo="y",
                legendgroup=f"group_{comb}",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=(
            dict(
                text=f"<b>Best Fitness Distribution across Run</b><br>Experience:{exp_comb}",
                x=0.5,
                xanchor="center",
            )
            if exp_comb
            else dict(
                text="<b>Best Fitness Distribution across Run</b>",
                x=0.5,
                xanchor="center",
            )
        ),
        xaxis=dict(
            title="Combination ID",
            gridcolor="lightgray",
            type="category",
            # tickmode='array',
            tickvals=combs,
        ),
        yaxis=dict(
            title="Best Fitness",
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        legend=dict(
            title=dict(text="Combinations", side="top", font=dict(size=12)),
            orientation="h",
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        autosize=True,
        height=600,
        width=850,
    )

    fig.show()


def comb_between_exp(df_exps: dict, comb_ids: list = None, metric="MBF", std=True):
    """
    Parameters:
    - df_exps: Dictionairy where the keys are the experience name (used in the comb_tuning funtion)
    and the values are the description of that experience.
    - comb_ids: Optional parameter that receives a list of combination ids ('ABC' where A, B and C are the
    initials of the mutation, crossover and selection functions, respectively) to only show for specific combinations.
    """

    # take just the unique combinations from any df indicated in df_exp
    any_exp = pd.read_csv(f"combination_search/final_results_{next(iter(df_exps))}.csv")
    if not comb_ids:
        comb_ids = any_exp["Combination ID"].unique()

    for comb in comb_ids:
        same_comb = []
        for exp_name, description in df_exps.items():
            df_exp = pd.read_csv(f"combination_search/final_results_{exp_name}.csv")
            df_one_comb_one_exp = df_exp[df_exp["Combination ID"] == comb].copy()
            # identify the experience by the description
            df_one_comb_one_exp["Experience"] = [
                description for _ in range(df_one_comb_one_exp.shape[0])
            ]

            same_comb.append(df_one_comb_one_exp)
        df_one_comb_all_exp = pd.concat(same_comb).reset_index(drop=True)

        combination = df_one_comb_all_exp["Combination"][0]
        plot_ABF_MBF(
            df_one_comb_all_exp,
            metric=metric,
            std=std,
            plot_by_exp=True,
            exp_comb=combination,
        )


def exp_bf_per_comb(df_exps: dict):
    any_exp = pd.read_csv(f"combination_search/final_results_{next(iter(df_exps))}.csv")
    comb_ids = any_exp["Combination ID"].unique()
    fig = go.Figure()

    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f"combination_search/final_results_{exp_name}.csv")
        # for each experiment get the maximum fitness values for each combination
        max_fitness = (
            df_exp.groupby(["Combination", "Combination ID"])["Fitness"]
            .max()
            .reset_index()
        )

        fig.add_trace(
            go.Scatter(
                x=max_fitness["Combination ID"],
                y=max_fitness["Fitness"],
                mode="markers",
                name=description,  # use the description as the trace name
                marker=dict(size=10),
                hovertemplate=(
                    "<b>Combination ID</b>: %{x}<br>"
                    "<b>Combination</b>: " + max_fitness["Combination"] + "<br>"
                    "<b>Fitness</b>: %{y:.3f}<br>"
                    "<b>Experiment</b>: " + description + "<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="<b>Best Fitness per Combination Across Experiments</b>",
        xaxis=dict(
            title="Combination ID",
            gridcolor="lightgray",
            type="category",
            tickmode="array",
            tickvals=comb_ids,
            ticktext=[str(c) for c in comb_ids],
        ),
        yaxis=dict(title="Best Fitness", gridcolor="lightgray"),
        plot_bgcolor="white",
        legend=dict(
            # title=dict(text='Experiments', side='top'),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=50, b=80, t=80, pad=10),
        height=600,
        width=900,
    )

    fig.show()


def comb_bf_per_exp(df_exps: dict):
    any_exp = pd.read_csv(f"combination_search/final_results_{next(iter(df_exps))}.csv")
    comb_ids = any_exp["Combination ID"].unique()
    fig = go.Figure()

    all_data = []
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f"combination_search/final_results_{exp_name}.csv")
        # for each experiment get the maximum fitness values for each combination
        max_fitness = (
            df_exp.groupby(["Combination", "Combination ID"])["Fitness"]
            .max()
            .reset_index()
        )
        max_fitness["Experiment"] = [description for _ in range(max_fitness.shape[0])]
        all_data.append(max_fitness)

    combined_df = pd.concat(all_data)

    for comb_id in comb_ids:
        comb_data = combined_df[combined_df["Combination ID"] == comb_id]
        comb_name = comb_data["Combination"].iloc[0]  # Get the combination name

        # First add the line trace (this will connect the dots)
        fig.add_trace(
            go.Scatter(
                x=comb_data["Experiment"],
                y=comb_data["Fitness"],
                mode="lines+markers",  # This adds both lines and markers
                name=f"Comb {comb_id}",
                line=dict(color="gray", width=1),  # Customize line appearance
                marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
                hovertemplate=(
                    "<b>Exp</b>: %{x}<br>"
                    "<b>Comb</b>: %{meta[1]}<br>"
                    "<b>Fitness</b>: %{y:.3f}<extra></extra>"
                ),
                meta=[comb_id, comb_name],
                showlegend=False,
            )
        )

        # Then add the marker trace (for better hover and legend appearance)
        fig.add_trace(
            go.Scatter(
                x=comb_data["Experiment"],
                y=comb_data["Fitness"],
                mode="markers",
                name=f"Comb {comb_id}",
                marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
                hovertemplate=(
                    "<b>Exp</b>: %{x}<br>"
                    "<b>Comb</b>: %{meta[1]}<br>"
                    "<b>Fitness</b>: %{y:.3f}<extra></extra>"
                ),
                meta=[comb_id, comb_name],
                showlegend=False,
            )
        )

    fig.update_layout(
        title="<b>Fitness of all Combinations per Experiments</b>",
        xaxis=dict(
            title="Experiments",
            gridcolor="lightgray",
            tickvals=list(range(len(df_exps))),  # Numeric positions (0, 1, 2, ...)
            ticktext=list(df_exps.values()),  # Experiment names as labels
            range=[-0.5, len(df_exps) - 0.5],  # Tighten the x-axis range
            showgrid=True,
        ),
        yaxis=dict(title="Fitness Value", gridcolor="lightgray"),
        plot_bgcolor="white",
        showlegend=True,  # Now show the legend
        legend=dict(
            title=dict(text="Combinations"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            # x=1
        ),
    )

    fig.show()


def get_abf_stats(df):
    """
    Computes the Average Best Fitness (ABF), standard deviation, and ABF-to-STD ratio
    for each combination based on the best fitness per run.

    Parameters:
    - df: The input Dataset.

    Returns:
    - Dataset with ABF, standard deviation, and ABF/STD ratio per combination.
    """

    fitness_col = "Fitness"
    comb_col = "Combination"
    run_col = "Run"

    if not {comb_col, run_col, fitness_col}.issubset(df.columns):
        raise ValueError("Missing required columns in the input DataFrame.")

    # Compute max fitness per combination per run
    best_per_run = df.groupby([comb_col, run_col])[fitness_col].max().reset_index()

    # Compute mean and std for each combination
    abf_mean = best_per_run.groupby(comb_col)[fitness_col].mean().rename("ABF")
    abf_std = best_per_run.groupby(comb_col)[fitness_col].std().rename("STD")

    # Combine into single DataFrame
    abf_stats = pd.concat([abf_mean, abf_std], axis=1).reset_index()
    abf_stats["ABF/STD"] = abf_stats["ABF"] / abf_stats["STD"]
    abf_stats["ABF^2/STD - ABF"] = (abf_stats["ABF"] ** 2) / abf_stats[
        "STD"
    ] - abf_stats["ABF"]

    idx_best_stability, idx_low_stability = (
        abf_stats["ABF^2/STD - ABF"].argmax(),
        abf_stats["ABF^2/STD - ABF"].argmin(),
    )
    best_stability = abf_stats.loc[idx_best_stability, "Combination"]
    low_stability = abf_stats.loc[idx_low_stability, "Combination"]

    styled_stats = abf_stats.style.apply(
        lambda x: [
            (
                "background: blue"
                if x.name == idx_best_stability
                else "background: brown" if x.name == idx_low_stability else ""
            )
            for _ in x
        ],
        axis=1,
    )

    print(f"Most Stable Comb: {best_stability}")
    print(f"Least Stable Comb: {low_stability}")

    return styled_stats, abf_stats


# Assuming best fitness is at the 3rd column
def elitism_tuning_plot(df):
    df_true = df[df["Elitism"] == True]
    df_false = df[df["Elitism"] == False]
    final_fitness_per_run_T = []
    final_fitness_per_run_F = []
    for run in range(1, df["Run"].max() + 1):
        df_t_runx = df_true[df_true["Run"] == run]
        ff_t_runx = df_t_runx.iloc[-1, 3]
        df_f_runx = df_false[df_false["Run"] == run]
        ff_f_runx = df_f_runx.iloc[-1, 3]
        final_fitness_per_run_T.append(ff_t_runx)
        final_fitness_per_run_F.append(ff_f_runx)

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=final_fitness_per_run_T,
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            line=dict(color="orange"),
            name="With Elitism",
        )
    )

    fig.add_trace(
        go.Box(
            y=final_fitness_per_run_F,
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            line=dict(color="orange"),
            name="Without Elitism",
        )
    )

    fig.update_layout(
        title=f"Impact on the use of Elitism (Number of Runs: {df['Run'].max()})",
        xaxis_title="",
        yaxis_title="Final Fitness",
        height=500,
        width=1100,
        yaxis_range=[0, None],
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        template="plotly_white",
    )

    fig.show()


def plot_Xbest_per_exp(df_exps: dict, Xbest: int = 4, metric="MBF", std=False):

    all_df_exps = []
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f"combination_search/final_results_{exp_name}.csv")
        df_exp["Experience Name"] = [exp_name for _ in range(df_exp.shape[0])]
        df_exp["Experience Description"] = [description for _ in range(df_exp.shape[0])]
        all_df_exps.append(df_exp)
    xbf_df = pd.concat(all_df_exps)

    if metric == "MBF":
        by_comb_exp = (
            xbf_df.groupby(
                [
                    "Combination",
                    "Combination ID",
                    "Experience Name",
                    "Experience Description",
                    "Generation",
                ]
            )["Fitness"]
            .median()
            .reset_index()
        )
    if metric == "ABF":
        by_comb_exp = (
            xbf_df.groupby(
                [
                    "Combination",
                    "Combination ID",
                    "Experience Name",
                    "Experience Description",
                    "Generation",
                ]
            )["Fitness"]
            .mean()
            .reset_index()
        )

    if std:
        std_df = (
            xbf_df.groupby(
                [
                    "Combination",
                    "Combination ID",
                    "Experience Name",
                    "Experience Description",
                    "Generation",
                ]
            )["Fitness"]
            .std()
            .reset_index()
        )
        std_df.rename(columns={"Fitness": "std"}, inplace=True)

        by_comb_exp = pd.merge(
            by_comb_exp,
            std_df,
            on=[
                "Combination",
                "Combination ID",
                "Experience Name",
                "Experience Description",
                "Generation",
            ],
        )

        by_comb_exp["y_upper"] = by_comb_exp["Fitness"] + by_comb_exp["std"]
        by_comb_exp["y_lower"] = by_comb_exp["Fitness"] - by_comb_exp["std"]
        by_comb_exp.loc[by_comb_exp["y_lower"] < 0, "y_lower"] = 0

    # CHOOSE ONLY X BEST combinations for each experiment
    # filter for the final fitness (last generation fitness)
    final_all_exp = by_comb_exp[
        by_comb_exp["Generation"] == by_comb_exp["Generation"].max()
    ]
    top_combinations = []
    for exp in df_exps.keys():
        # filter only by this experiment
        exp_data = final_all_exp[final_all_exp["Experience Name"] == exp]
        # get top X combinations by fitness
        top_X = exp_data.nlargest(Xbest, "Fitness")
        # store the combination identifiers
        for _, row in top_X.iterrows():
            top_combinations.append(
                {
                    "Combination ID": row["Combination ID"],
                    "Experience Name": row["Experience Name"],
                }
            )

    # filter the full data to only include top combinations
    filtered_data = []
    for comb in top_combinations:
        mask = (by_comb_exp["Combination ID"] == comb["Combination ID"]) & (
            by_comb_exp["Experience Name"] == comb["Experience Name"]
        )
        filtered_data.append(by_comb_exp[mask])

    plot_df = pd.concat(filtered_data)

    # Create the plot
    fig = go.Figure()

    # Group by combination to plot each one
    grouped = plot_df.groupby(
        ["Combination ID", "Combination", "Experience Name", "Experience Description"]
    )
    for (comb_id, comb, exp_name, exp_desc), group in grouped:
        fig.add_trace(
            go.Scatter(
                x=group["Generation"],
                y=group["Fitness"],
                mode="lines+markers",
                name=f"Comb {comb_id}: {comb},<br>{exp_name}: {exp_desc}",
                hovertemplate=f"Combination: {comb}<br>Experiment: {exp_name}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>",
            )
        )
        if std:
            fig.add_trace(
                go.Scatter(
                    x=group["Generation"],
                    y=group["y_upper"],
                    mode="lines",
                    name="+1 std Train",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group["Generation"],
                    y=group["y_lower"],
                    mode="lines",
                    name="-1 std Train",
                    fill="tonexty",
                    fillcolor="rgba(0,0,255,0.1)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Best fitness along the generation from the <b>{Xbest}</b> best of each experiment",
        title_font=dict(family="Arial", size=18),
        xaxis=dict(title="Generation", gridcolor="lightgray"),
        yaxis=dict(title="Fitness", gridcolor="lightgray"),
        plot_bgcolor="white",
        legend=dict(
            title=dict(text="Combination & Experiment", side="top", font=dict(size=12)),
            orientation="h",
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        height=650,
        width=850,
    )

    fig.show()


def plot_Xbest_per_exp_box(df_exps: dict, Xbest: int = 4, metric="MBF"):

    all_df_exps = []
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f"combination_search/final_results_{exp_name}.csv")
        df_exp["Experience Name"] = [exp_name for _ in range(df_exp.shape[0])]
        df_exp["Experience Description"] = [description for _ in range(df_exp.shape[0])]
        all_df_exps.append(df_exp)
    xbf_df = pd.concat(all_df_exps)

    if metric == "MBF":
        by_comb_exp = (
            xbf_df.groupby(
                [
                    "Combination",
                    "Combination ID",
                    "Experience Name",
                    "Experience Description",
                    "Generation",
                ]
            )["Fitness"]
            .median()
            .reset_index()
        )
    if metric == "ABF":
        by_comb_exp = (
            xbf_df.groupby(
                [
                    "Combination",
                    "Combination ID",
                    "Experience Name",
                    "Experience Description",
                    "Generation",
                ]
            )["Fitness"]
            .mean()
            .reset_index()
        )

    # CHOOSE ONLY X BEST combinations for each experiment
    # filter for the final fitness (last generation fitness)
    final_all_exp = by_comb_exp[
        by_comb_exp["Generation"] == by_comb_exp["Generation"].max()
    ]
    top_combinations = []
    for exp in df_exps.keys():
        # filter only by this experiment
        exp_data = final_all_exp[final_all_exp["Experience Name"] == exp]
        # get top X combinations by fitness
        top_X = exp_data.nlargest(Xbest, "Fitness")
        # store the combination identifiers
        for _, row in top_X.iterrows():
            top_combinations.append(
                {
                    "Combination ID": row["Combination ID"],
                    "Experience Name": row["Experience Name"],
                }
            )

    # filter the full data to only include top combinations
    filtered_data = []
    for comb in top_combinations:
        mask = (by_comb_exp["Combination ID"] == comb["Combination ID"]) & (
            by_comb_exp["Experience Name"] == comb["Experience Name"]
        )
        filtered_data.append(by_comb_exp[mask])

    plot_df = pd.concat(filtered_data)

    # Create the plot
    fig = go.Figure()

    # Group by combination to plot each one
    grouped = plot_df.groupby(
        ["Combination ID", "Combination", "Experience Name", "Experience Description"]
    )
    unique_labels = []
    for (comb_id, comb, exp_name, exp_desc), group in grouped:
        fig.add_trace(
            go.Box(
                y=group["Fitness"],
                # name = f'{comb_id}<br>{exp_name}',
                name=f"Comb {comb_id}: {comb},<br>{exp_name}: {exp_desc}",
                # x=group,
                marker=dict(size=5),
                line=dict(width=1),
                hoverinfo="y",
                # hovertemplate=f'Combination: {comb}<br>Experiment: {exp_name}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>'
                legendgroup=f"group_{comb}",
                showlegend=True,
            )
        )
        unique_labels.append((f"{comb_id}<br>{exp_name}"))

    fig.update_layout(
        title=f"Best fitness along the generation from the <b>{Xbest}</b> best of each experiment",
        title_font=dict(family="Arial", size=18),
        # xaxis=dict(title='Generation', gridcolor='lightgray'),
        yaxis=dict(title="Fitness", gridcolor="lightgray"),
        xaxis=dict(
            gridcolor="lightgray",
            type="category",
            tickvals=list(range(len(unique_labels))),
            ticktext=unique_labels,
        ),
        plot_bgcolor="white",
        legend=dict(
            title=dict(text="Combination & Experiment", side="top", font=dict(size=12)),
            orientation="h",
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        height=650,
        width=850,
    )

    fig.show()


def perform_elitism_mannwhitneyu_test(df):
    """
    Performs a Mann-Whitney U test on the fitness of runs in their final generation,
    comparing groups with and without elitism.

    Args:
        df (pd.DataFrame): DataFrame containing 'Elitism', 'Generation', and 'Fitness' columns.
    """
    max_generation = df["Generation"].max()

    last_gen_true = df[(df["Elitism"] == True) & (df["Generation"] == max_generation)]
    true_fitness = last_gen_true["Fitness"]

    last_gen_false = df[(df["Elitism"] == False) & (df["Generation"] == max_generation)]
    false_fitness = last_gen_false["Fitness"]

    if true_fitness.empty or false_fitness.empty:
        print(
            f"Not enough data for one or both groups at generation {max_generation} to perform the test."
        )
        return

    u_stat, p_value = mannwhitneyu(true_fitness, false_fitness, alternative="two-sided")

    # Calculate medians for reporting
    true_median = true_fitness.median()
    false_median = false_fitness.median()

    # Display results
    report_title = (
        f"Mann-Whitney U Test: Elitism (True vs. False) - Generation {max_generation}"
    )
    print(f"\n{report_title}")
    print("-" * len(report_title))  # Underline for the title

    label_width = 35  # Width for aligning labels

    # Using f-string alignment for a cleaner, table-like look
    print(f"  {'U statistic':<{label_width}}: {u_stat}")
    print(
        f"  {'P-value':<{label_width}}: {p_value:.4g}"
    )  # Format p-value to 4 significant figures
    print(f"  {'Median Fitness (with Elitism)':<{label_width}}: {true_median:.5f}")
    print(f"  {'Median Fitness (without Elitism)':<{label_width}}: {false_median:.5f}")

    alpha = 0.05
    significance_message = "Yes" if p_value < alpha else "No"
    # Using the Greek letter alpha (α) for the significance level
    print(
        f"  {'Significant difference (α=0.05)':<{label_width}}: {significance_message}"
    )

    print("-" * len(report_title) + "\n")  # Footer line


def pairwise_mannwhitney(
    combs: List[Tuple], alpha: float = 0.05, plot: bool = True
) -> Dict[str, Any]:
    """
    Perform pairwise Mann-Whitney U tests on the provided combinations and visualize results.

    Parameters:
    - combs: List of tuples containing combinations to compare.
    - alpha: Significance level for the tests.
    - plot: Whether to generate a visualization of the results.

    Returns:
    - Dictionary with combinations as keys and p-values as values.
    """

    df_exp1 = pd.read_csv("combination_search/final_results_exp1.csv")
    df_exp2 = pd.read_csv("combination_search/final_results_exp2.csv")
    df_exp3 = pd.read_csv("combination_search/final_results_exp3.csv")

    data_combs = []
    labels = []

    for comb, exp in combs:
        if exp == "exp1":
            df = df_exp1
        elif exp == "exp2":
            df = df_exp2
        else:
            df = df_exp3

        # Get the last generation fitness values for all runs of this combination
        data = df[
            (df["Combination ID"] == comb)
            & (df["Generation"] == df["Generation"].max())
        ]["Fitness"].values
        data_combs.append(data)
        labels.append(f"{comb}({exp})")

    # Perform pairwise Mann-Whitney U tests
    n = len(combs)
    p_values = np.zeros((n, n))
    significant = np.zeros((n, n))

    results = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                p_values[i, j] = 1.0
                continue

            stat, p_value = mannwhitneyu(
                data_combs[i], data_combs[j], alternative="two-sided"
            )
            p_values[i, j] = p_value
            significant[i, j] = p_value < alpha

            key = f"{labels[i]} vs {labels[j]}"
            results[key] = {
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < alpha,
            }

    if plot:
        plt.figure(figsize=(6, 5))

        # Create custom color palette
        cmap = sns.color_palette("YlOrRd_r", as_cmap=True)

        # Create heatmap for p-values
        mask = np.triu(np.ones_like(p_values, dtype=bool))

        # Format annotations to show significance level with stars
        annot = np.empty_like(p_values, dtype=object)
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    annot[i, j] = ""
                else:
                    stars = ""
                    if p_values[i, j] < 0.001:
                        stars = "***"
                    elif p_values[i, j] < 0.01:
                        stars = "**"
                    elif p_values[i, j] < 0.05:
                        stars = "*"
                    annot[i, j] = f"{p_values[i, j]:.3f}{stars}"

        ax = sns.heatmap(
            p_values,
            annot=annot,
            fmt="",
            mask=mask,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=alpha * 2,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "p-value"},
        )

        # Add grid for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # # Add border to significant cells
        # for i in range(n):
        #     for j in range(n):
        #         if i != j and not mask[i, j] and significant[i, j]:
        #             ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=2, linestyle='--'))

        plt.title(
            f"Pairwise Mann-Whitney U Test p-values\n*p<0.05, **p<0.01, ***p<0.001",
            fontsize=14,
        )
        plt.tight_layout()

        plt.show()

    return results
