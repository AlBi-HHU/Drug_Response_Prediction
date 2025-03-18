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
    for i, s in enumerate(solver[::-1]):
        scores = []
        for j in range(n_splits):
            with open(snakemake.input[::-1][i*n_splits + j], 'r') as f:
                score = float(f.readline())
            scores.append(score)
        scores = np.array(scores)
        score_info[s] = [np.nanmean(scores), np.nanstd(scores)]
        fig.add_trace(
            go.Box(
                name=s,
                x=scores.flatten(),
                marker_color=colors[i%len(colors)],
                showlegend=False,
                boxmean='sd'
            )
        )
    """while i < 13:
        fig.add_trace(
            go.Box(
                name=s + str(i),
                x=scores.flatten(),
                marker_color=colors[i%len(colors)],
                showlegend=False,
                boxmean='sd'
            )
        )
        i += 1""" # was only needed for same layout for manuscript

    fig.update_layout(
        autosize=False,
        title=title,
        title_x=0.5,
        title_y=0.98,
        xaxis_title=metric,
        margin=dict(
            l=110,
            r=10,
            t=30,
            b=0
        )
    )

    plotly.io.write_image(fig, snakemake.output[0], format='pdf')

    with open(snakemake.output[0][:-4] + '.json', 'w') as f:
        json.dump(score_info, f)
