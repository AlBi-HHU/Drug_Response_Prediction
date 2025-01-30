import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io

if __name__ == "__main__":
    # plot each metric score of each output for each split
    solver = snakemake.config['solver']
    n_splits = snakemake.config['n_splits']
    resampling = snakemake.wildcards['resampling']
    metric = snakemake.wildcards['metric']
    
    """title = 'Record-wise'
    if resampling == 'subjectwise':
        title = 'Subject-wise'
    if resampling == 'targetwise':
        title = 'Target-wise'"""
    title = 'LPO'
    if resampling == 'subjectwise':
        title = 'LCO'
    if resampling == 'targetwise':
        title = 'LDO'
    title += f' {n_splits}-fold CV' #<br>{snakemake.wildcards["inputs"]}'

    colors = ['#006ab3']

    score_info = {}
    fig = go.Figure()
    for i, s in enumerate(solver):
        scores = []
        for j in range(n_splits):
            with open(snakemake.input[i*n_splits + j], 'r') as f:
                score = float(f.readline())
            scores.append(score)
        scores = np.array(scores)
        score_info[s] = [np.nanmean(scores), np.nanstd(scores)]
        fig.add_trace(
            go.Box(
                name=s,
                y=scores.flatten(),
                marker_color=colors[i%len(colors)],
                showlegend=False,
                boxmean='sd'
            )
        )

    fig.update_layout(
        title=title,
        title_x=0.5,
        title_y=0.85,
        yaxis_title=metric,
        xaxis_tickangle=90
    )
    #if resampling == 'targetwise':
    #    fig.update_layout(width=400)

    plotly.io.write_image(fig, snakemake.output[0], format='pdf')

    with open(snakemake.output[0][:-4] + '.json', 'w') as f:
        json.dump(score_info, f)
