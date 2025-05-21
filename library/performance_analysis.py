import plotly.graph_objects as go
import pandas as pd

def plot_ABF_MBF(df, metric, std=False, exp_comb=None, plot_by_exp=False):

    """
    Plots average/median best fitness per generation for each configuration.

    Parameters:
    - df: Dataset containing fitness data
    - metric: receives either 'ABF' (Average Best Fitness) or 'MBF' (Median Best Fitness) as arguments
    - std: If True it plots conjointly in the plot
    - exp_comb: To set the specific experience/combination in the title

        Specific Parameters to change only when calling this function inside the comb_between_exp function:
            - plot_by_exp: Change to True to plot based on different experiments
    """

    if plot_by_exp:
        comb_col = 'Experience'
        title_col = 'Combination'
    else:
        comb_col = 'Combination'
        title_col = 'Experience'

    if metric == 'ABF':
        xbf_df = df.groupby([comb_col, 'Generation'])['Fitness'].mean().reset_index()
        title = f'<b>Average Best Fitness per Generation</b><br>{title_col} {exp_comb}' 
    if metric == 'MBF':
        xbf_df = df.groupby([comb_col, 'Generation'])['Fitness'].median().reset_index()
        title = f'<b>Median Best Fitness per Generation</b><br>{title_col} {exp_comb}'

    if std:
        # it is equal either plotting mbf or abf        
        std_df = df.groupby([comb_col, 'Generation'])['Fitness'].std().reset_index()
        std_df.rename(columns={'Fitness': 'std'}, inplace=True)
        std_col = std_df[['std']].copy()
        xbf_df = pd.concat([xbf_df, std_col], axis=1) # adding the column for the fitness std

        xbf_df['y_upper'] = xbf_df['Fitness'] + xbf_df['std']
        xbf_df['y_lower'] = xbf_df['Fitness'] - xbf_df['std']
        xbf_df.loc[xbf_df['y_lower'] < 0, 'y_lower'] = 0

    combs = df[comb_col].unique()

    fig = go.Figure()

    for comb in combs:
        data = xbf_df[xbf_df[comb_col] == comb]
        y_vals = data['Fitness'].values
        
        fig.add_trace(go.Scatter(
            x=xbf_df['Generation'].values,
            y=y_vals,
            mode='lines+markers',
            name=str(comb),
            showlegend=True,
            hovertemplate=f'{comb_col}: {comb}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>'
        ))
        if std:
            fig.add_trace(go.Scatter(
                x=xbf_df['Generation'].values,
                y=data['y_upper'],
                mode='lines',
                name='+1 std Train',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=xbf_df['Generation'],
                y=data['y_lower'],
                mode='lines',
                name='-1 std Train',
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(width=0),
                showlegend=False
            ))

    fig.update_layout(
        title=title,
        title_font=dict(family="Arial", size=18),
        xaxis=dict(
            title='Generation',
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title='Fitness',
            gridcolor='lightgray',
        ),
        plot_bgcolor='white',
        legend=dict(
            title=dict(text=f'{comb_col}',
                     side='top',  # positions title above the legend items
                     font=dict(size=12)),
            orientation='h',
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        autosize=True,
        height=650,
        width=850,
    )

    fig.show()


def plot_BF_inter_run(df, comb_ids:list=None, exp_comb:str=None):

    """
    Plots best fitness per run for each configuration.
    The objective is to evaluate the stability of the combination by seeing if 
    it maintains the results consistenly.

    Parameters:
    - df: Dataset containing fitness data
    """

    bf_run = df.groupby(['Combination', 'Run'])['Fitness'].max().reset_index()
    if comb_ids:
        comb_id_pair = df[['Combination', 'Combination ID']].drop_duplicates()
        comb_id_pair_filtered = comb_id_pair[comb_id_pair['Combination ID'].isin(comb_ids)]
        combs = comb_id_pair_filtered['Combination'].unique()
    else:
        combs = df['Combination'].unique()

    fig = go.Figure()

    for comb in combs:
        data = bf_run[bf_run['Combination'] == comb]
        y_vals = data['Fitness'].values
        
        fig.add_trace(go.Scatter(
            x=bf_run['Run'].values,
            y=y_vals,
            mode='lines+markers',
            name=str(comb),
            hovertemplate=f'Combination: {comb}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=f'<b>Best Fitness per Run</b><br>Experience:{exp_comb}', x=0.5, xanchor='center') if exp_comb 
        else dict(text='<b>Best Fitness per Run</b>', x=0.5, xanchor='center'),
        xaxis=dict(
            title='Run',
            gridcolor='lightgray',
            tickmode='linear',  # Ensures ticks are shown at regular intervals
            dtick=1,
            range=[0, None]
        ),
        yaxis=dict(
            title='Best Fitness',
            gridcolor='lightgray',
        ),
        plot_bgcolor='white',
        legend=dict(
            title=dict(text='Combinations',  # Added bold for emphasis
                     side='top',  # Positions title above the legend items
                     font=dict(size=12)),
            orientation='h',
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        autosize=True,
        height=650,
        width=850,
    )

    fig.show()

def plot_BF_inter_run_boxplot(df, comb_ids:list=None, exp_comb:str=None):
    """
    Plots boxplots of best fitness per run for each configuration.
    The objective is to evaluate the stability and distribution of results across runs.

    Parameters:
    - df: Dataset containing fitness data
    - comb_ids: List of combination IDs to filter by (optional)
    - exp_comb: Experiment combination name for title (optional)
    """
    
    # get the best fitness per run for each combination
    bf_run = df.groupby(['Combination', 'Combination ID', 'Run'])['Fitness'].max().reset_index()
    
    # filter combinations if IDs are provided
    if comb_ids:
        bf_run = bf_run[bf_run['Combination ID'].isin(comb_ids)]
    
    fig = go.Figure()

    # get unique combinations
    combs = bf_run['Combination ID'].unique()
    
    for comb in combs:
        data = bf_run[bf_run['Combination ID'] == comb]
        comb_str = data[data['Combination ID'] == comb][['Combination']].iloc[0, 0]

        fig.add_trace(go.Box(
            y=data['Fitness'],
            name=f"{comb}: {comb_str}",
            x=[comb]*len(data),
            marker=dict(size=5),
            line=dict(width=1),
            hoverinfo='y',
            legendgroup=f"group_{comb}",
            showlegend=True
        ))

    fig.update_layout(
        title=dict(text=f'<b>Best Fitness Distribution across Run</b><br>Experience:{exp_comb}', x=0.5, xanchor='center') if exp_comb 
        else dict(text='<b>Best Fitness Distribution across Run</b>', x=0.5, xanchor='center'),
        xaxis=dict(
            title='Combination ID',
            gridcolor='lightgray',
            type='category',
            #tickmode='array',
            tickvals=combs,
        ),
        yaxis=dict(
            title='Best Fitness',
            gridcolor='lightgray',
        ),
        plot_bgcolor='white',
        legend=dict(
            title=dict(text='Combinations',
                     side='top',
                     font=dict(size=12)),
            orientation='h',
            y=-0.3,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        autosize=True,
        height=600,
        width=850,
    )

    fig.show()


def comb_between_exp(df_exps:dict, comb_ids:list = None, metric='MBF', std=True):
    """
    Parameters:
    - df_exps: Dictionairy where the keys are the experience name (used in the comb_tuning funtion)
    and the values are the description of that experience.
    - comb_ids: Optional parameter that receives a list of combination ids ('ABC' where A, B and C are the 
    initials of the mutation, crossover and selection functions, respectively) to only show for specific combinations.
    """

    # take just the unique combinations from any df indicated in df_exp
    any_exp = pd.read_csv(f'combination_search/final_results_{next(iter(df_exps))}.csv')
    if not comb_ids:
        comb_ids = any_exp['Combination ID'].unique()

    for comb in comb_ids:
        same_comb = []
        for exp_name, description in df_exps.items():
            df_exp = pd.read_csv(f'combination_search/final_results_{exp_name}.csv')
            df_one_comb_one_exp = df_exp[df_exp['Combination ID']==comb].copy()
            # identify the experience by the description
            df_one_comb_one_exp['Experience'] = [description for _ in range(df_one_comb_one_exp.shape[0])] 
            
            same_comb.append(df_one_comb_one_exp)
        df_one_comb_all_exp = pd.concat(same_comb).reset_index(drop=True)

        combination = df_one_comb_all_exp['Combination'][0]
        plot_ABF_MBF(df_one_comb_all_exp, metric=metric, std=std, plot_by_exp=True, exp_comb=combination)

def exp_bf_per_comb(df_exps:dict):
    any_exp = pd.read_csv(f'combination_search/final_results_{next(iter(df_exps))}.csv')
    comb_ids = any_exp['Combination ID'].unique()
    fig = go.Figure()
    
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f'combination_search/final_results_{exp_name}.csv')
        #for each experiment get the maximum fitness values for each combination
        max_fitness = df_exp.groupby(['Combination', 'Combination ID'])['Fitness'].max().reset_index()

        fig.add_trace(go.Scatter(
            x=max_fitness['Combination ID'],
            y=max_fitness['Fitness'],
            mode='markers',
            name=description,  # use the description as the trace name
            marker=dict(size=10),
            hovertemplate=(
                "<b>Combination ID</b>: %{x}<br>"
                "<b>Combination</b>: " + max_fitness['Combination'] + "<br>"
                "<b>Fitness</b>: %{y:.3f}<br>"
                "<b>Experiment</b>: " + description + "<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title='<b>Best Fitness per Combination Across Experiments</b>',
        xaxis=dict(
            title='Combination ID',
            gridcolor='lightgray',
            type='category',
            tickmode='array',
            tickvals=comb_ids,
            ticktext=[str(c) for c in comb_ids]
        ),
        yaxis=dict(
            title='Best Fitness',
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        legend=dict(
            #title=dict(text='Experiments', side='top'),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, b=80, t=80, pad=10),
        height=600,
        width=900
    )

    fig.show()

def comb_bf_per_exp(df_exps: dict):
    any_exp = pd.read_csv(f'combination_search/final_results_{next(iter(df_exps))}.csv')
    comb_ids = any_exp['Combination ID'].unique()
    fig = go.Figure()
    
    all_data = []
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f'combination_search/final_results_{exp_name}.csv')
        # for each experiment get the maximum fitness values for each combination
        max_fitness = df_exp.groupby(['Combination', 'Combination ID'])['Fitness'].max().reset_index()
        max_fitness['Experiment'] = [description for _ in range(max_fitness.shape[0])]
        all_data.append(max_fitness)

    combined_df = pd.concat(all_data)

    for comb_id in comb_ids:
        comb_data = combined_df[combined_df['Combination ID'] == comb_id]
        comb_name = comb_data['Combination'].iloc[0]  # Get the combination name
        
        # First add the line trace (this will connect the dots)
        fig.add_trace(go.Scatter(
            x=comb_data['Experiment'],
            y=comb_data['Fitness'],
            mode='lines+markers',  # This adds both lines and markers
            name=f"Comb {comb_id}",
            line=dict(color='gray', width=1),  # Customize line appearance
            marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate=(
                "<b>Exp</b>: %{x}<br>"
                "<b>Comb</b>: %{meta[1]}<br>"
                "<b>Fitness</b>: %{y:.3f}<extra></extra>"
            ),
            meta=[comb_id, comb_name],
            showlegend=False
        ))
        
        # Then add the marker trace (for better hover and legend appearance)
        fig.add_trace(go.Scatter(
            x=comb_data['Experiment'],
            y=comb_data['Fitness'],
            mode='markers',
            name=f"Comb {comb_id}",
            marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate=(
                "<b>Exp</b>: %{x}<br>"
                "<b>Comb</b>: %{meta[1]}<br>"
                "<b>Fitness</b>: %{y:.3f}<extra></extra>"
            ),
            meta=[comb_id, comb_name],
            showlegend=False,
        ))

    fig.update_layout(
        title='<b>Fitness of all Combinations per Experiments</b>',
        xaxis=dict(
            title='Experiments',
            gridcolor='lightgray',
            tickvals=list(range(len(df_exps))),  # Numeric positions (0, 1, 2, ...)
            ticktext=list(df_exps.values()),  # Experiment names as labels
            range=[-0.5, len(df_exps) - 0.5],  # Tighten the x-axis range
            showgrid=True,
        ),
        yaxis=dict(
            title='Fitness Value',
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        showlegend=True,  # Now show the legend
        legend=dict(
            title=dict(text='Combinations'),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            #x=1
        )
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

    fitness_col='Fitness'
    comb_col='Combination'
    run_col='Run'
    
    if not {comb_col, run_col, fitness_col}.issubset(df.columns):
        raise ValueError("Missing required columns in the input DataFrame.")

    # Compute max fitness per combination per run
    best_per_run = df.groupby([comb_col, run_col])[fitness_col].max().reset_index()

    # Compute mean and std for each combination
    abf_mean = best_per_run.groupby(comb_col)[fitness_col].mean().rename('ABF')
    abf_std = best_per_run.groupby(comb_col)[fitness_col].std().rename('STD')

    # Combine into single DataFrame
    abf_stats = pd.concat([abf_mean, abf_std], axis=1).reset_index()
    abf_stats['ABF/STD'] = abf_stats['ABF'] / abf_stats['STD']
    abf_stats['ABF^2/STD - ABF'] = (abf_stats['ABF']**2) / abf_stats['STD'] - abf_stats['ABF']

    idx_best_stability, idx_low_stability = abf_stats['ABF/STD'].argmax(), abf_stats['ABF/STD'].argmin()
    best_stability = abf_stats.loc[idx_best_stability, 'Combination']
    low_stability = abf_stats.loc[idx_low_stability, 'Combination']

    styled_stats = abf_stats.style.apply(
    lambda x: ['background: blue' if x.name == idx_best_stability
               else 'background: brown' if x.name == idx_low_stability else '' for _ in x],
    axis=1
)

    print(f'Most Stable Comb: {best_stability}')
    print(f'Least Stable Comb: {low_stability}')

    return styled_stats, abf_stats


#Assuming best fitness is at the 3rd column
def elistim_tuning_plot(df):
    df_true = df[df['Elitism']==True]
    df_false = df[df['Elitism']==False]
    final_fitness_per_run_T = []
    final_fitness_per_run_F = []
    for run in range(1, df['Run'].max()+1):
        df_t_runx = df_true[df_true['Run']==run]
        ff_t_runx = df_t_runx.iloc[-1, 3]
        df_f_runx = df_false[df_false['Run']==run]
        ff_f_runx = df_f_runx.iloc[-1, 3]
        final_fitness_per_run_T.append(ff_t_runx)
        final_fitness_per_run_F.append(ff_f_runx)

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=final_fitness_per_run_T,
        boxpoints='all',
        jitter=0.5,
        pointpos=0,
        line=dict(color='orange'),
        name='With Elitism'
    ))

    fig.add_trace(go.Box(
        y=final_fitness_per_run_F,
        boxpoints='all',
        jitter=0.5,
        pointpos=0,
        line=dict(color='orange'),
        name='Without Elitism'
    ))

    fig.update_layout(
        title=f'Impact on the use of Elitism (Number of Runs: {df['Run'].max()})',
        xaxis_title='',
        yaxis_title='Final Fitness',
        height=500, width=1100,
        yaxis_range=[0,None],
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        template='plotly_white'
    )

    fig.show()

def plot_Xbest_per_exp(df_exps:dict, Xbest: int = 4, metric='MBF', std=False):

    all_df_exps = []
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f'combination_search/final_results_{exp_name}.csv')
        df_exp['Experience Name'] = [exp_name for _ in range(df_exp.shape[0])]
        df_exp['Experience Description'] = [description for _ in range(df_exp.shape[0])]
        all_df_exps.append(df_exp)
    xbf_df = pd.concat(all_df_exps)

    if metric == 'MBF':
        by_comb_exp = xbf_df.groupby(['Combination', 'Combination ID', 'Experience Name', 'Experience Description',
                                'Generation'])['Fitness'].median().reset_index()
    if metric == 'ABF':
        by_comb_exp = xbf_df.groupby(['Combination', 'Combination ID', 'Experience Name', 'Experience Description',
                                'Generation'])['Fitness'].mean().reset_index()    

    if std:      
        std_df = xbf_df.groupby(['Combination', 'Combination ID', 'Experience Name', 'Experience Description',
                                'Generation'])['Fitness'].std().reset_index()
        std_df.rename(columns={'Fitness': 'std'}, inplace=True)

        by_comb_exp = pd.merge(by_comb_exp, std_df, 
                              on=['Combination', 'Combination ID', 'Experience Name', 'Experience Description', 'Generation'])
        
        by_comb_exp['y_upper'] = by_comb_exp['Fitness'] + by_comb_exp['std']
        by_comb_exp['y_lower'] = by_comb_exp['Fitness'] - by_comb_exp['std']
        by_comb_exp.loc[by_comb_exp['y_lower'] < 0, 'y_lower'] = 0                                                            
    
    #CHOOSE ONLY X BEST combinations for each experiment 
    # filter for the final fitness (last generation fitness)
    final_all_exp = by_comb_exp[by_comb_exp['Generation'] == by_comb_exp['Generation'].max()]
    top_combinations = []
    for exp in df_exps.keys():
        # filter only by this experiment
        exp_data = final_all_exp[final_all_exp['Experience Name'] == exp]
        # get top X combinations by fitness
        top_X = exp_data.nlargest(Xbest, 'Fitness')
        # store the combination identifiers
        for _, row in top_X.iterrows():
            top_combinations.append({
                'Combination ID': row['Combination ID'],
                'Experience Name': row['Experience Name']
            })
    
    # filter the full data to only include top combinations
    filtered_data = []
    for comb in top_combinations:
        mask = ((by_comb_exp['Combination ID'] == comb['Combination ID']) & 
                (by_comb_exp['Experience Name'] == comb['Experience Name']))
        filtered_data.append(by_comb_exp[mask])
    
    plot_df = pd.concat(filtered_data)
    
    # Create the plot
    fig = go.Figure()
    
    # Group by combination to plot each one
    grouped = plot_df.groupby(['Combination ID', 'Combination', 'Experience Name', 'Experience Description'])
    for (comb_id, comb, exp_name, exp_desc), group in grouped:
        fig.add_trace(go.Scatter(
            x=group['Generation'],
            y=group['Fitness'],
            mode='lines+markers',
            name=f"Comb {comb_id}: {comb},<br>{exp_name}: {exp_desc}",
            hovertemplate=f'Combination: {comb}<br>Experiment: {exp_name}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>'
        ))
        if std:
            fig.add_trace(go.Scatter(
                x=group['Generation'],
                y=group['y_upper'],
                mode='lines',
                name='+1 std Train',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=group['Generation'],
                y=group['y_lower'],
                mode='lines',
                name='-1 std Train',
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(width=0),
                showlegend=False
            ))
    
    fig.update_layout(
        title=f'Best fitness along the generation from the <b>{Xbest}</b> best of each experiment',
        title_font=dict(family="Arial", size=18),
        xaxis=dict(title='Generation', gridcolor='lightgray'),
        yaxis=dict(title='Fitness', gridcolor='lightgray'),
        plot_bgcolor='white',
        legend=dict(
            title=dict(text='Combination & Experiment', side='top', font=dict(size=12)),
            orientation='h',
            y=-0.3,
            font=dict(size=11)),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        height=650,
        width=850,
    )
    
    fig.show()


def plot_Xbest_per_exp_box(df_exps:dict, Xbest: int = 4, metric='MBF'):

    all_df_exps = []
    for exp_name, description in df_exps.items():
        df_exp = pd.read_csv(f'combination_search/final_results_{exp_name}.csv')
        df_exp['Experience Name'] = [exp_name for _ in range(df_exp.shape[0])]
        df_exp['Experience Description'] = [description for _ in range(df_exp.shape[0])]
        all_df_exps.append(df_exp)
    xbf_df = pd.concat(all_df_exps)

    if metric == 'MBF':
        by_comb_exp = xbf_df.groupby(['Combination', 'Combination ID', 'Experience Name', 'Experience Description',
                                'Generation'])['Fitness'].median().reset_index()
    if metric == 'ABF':
        by_comb_exp = xbf_df.groupby(['Combination', 'Combination ID', 'Experience Name', 'Experience Description',
                                'Generation'])['Fitness'].mean().reset_index()                                                            
    
    #CHOOSE ONLY X BEST combinations for each experiment 
    # filter for the final fitness (last generation fitness)
    final_all_exp = by_comb_exp[by_comb_exp['Generation'] == by_comb_exp['Generation'].max()]
    top_combinations = []
    for exp in df_exps.keys():
        # filter only by this experiment
        exp_data = final_all_exp[final_all_exp['Experience Name'] == exp]
        # get top X combinations by fitness
        top_X = exp_data.nlargest(Xbest, 'Fitness')
        # store the combination identifiers
        for _, row in top_X.iterrows():
            top_combinations.append({
                'Combination ID': row['Combination ID'],
                'Experience Name': row['Experience Name']
            })
    
    # filter the full data to only include top combinations
    filtered_data = []
    for comb in top_combinations:
        mask = ((by_comb_exp['Combination ID'] == comb['Combination ID']) & 
                (by_comb_exp['Experience Name'] == comb['Experience Name']))
        filtered_data.append(by_comb_exp[mask])
    
    plot_df = pd.concat(filtered_data)
    
    # Create the plot
    fig = go.Figure()
    
    # Group by combination to plot each one
    grouped = plot_df.groupby(['Combination ID', 'Combination', 'Experience Name', 'Experience Description'])
    unique_labels=[]
    for (comb_id, comb, exp_name, exp_desc), group in grouped:
        fig.add_trace(go.Box(
            y=group['Fitness'],
            #name = f'{comb_id}<br>{exp_name}',
            name=f"Comb {comb_id}: {comb},<br>{exp_name}: {exp_desc}",
            #x=group,
            marker=dict(size=5),
            line=dict(width=1),
            hoverinfo='y',
            #hovertemplate=f'Combination: {comb}<br>Experiment: {exp_name}<br>Generation: %{{x}}<br>Fitness: %{{y}}<extra></extra>'
            legendgroup=f"group_{comb}",
            showlegend=True
        ))
        unique_labels.append((f'{comb_id}<br>{exp_name}'))
    
    fig.update_layout(
        title=f'Best fitness along the generation from the <b>{Xbest}</b> best of each experiment',
        title_font=dict(family="Arial", size=18),
        #xaxis=dict(title='Generation', gridcolor='lightgray'),
        yaxis=dict(title='Fitness', gridcolor='lightgray'),
        xaxis=dict(
            gridcolor='lightgray',
            type='category',
            tickvals=list(range(len(unique_labels))),
            ticktext=unique_labels
        ),
        plot_bgcolor='white',
        legend=dict(
            title=dict(text='Combination & Experiment', side='top', font=dict(size=12)),
            orientation='h',
            y=-0.3,
            font=dict(size=11)),
        margin=dict(l=50, r=50, b=100, t=80, pad=10),
        height=650,
        width=850,
    )
    
    fig.show()
