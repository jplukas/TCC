#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from utils import *

def plot_model_results(m, subf, title):

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs = subf.subplots(len(m["sizes"]) // 2, 2)

    for i, row in enumerate(m.itertuples()):
        (embedder, granularity), n_topics, outliers, max_cluster_size, n_neighbors, n_components, min_cluster_size,min_samples, sizes = row
        if -1 in sizes: del sizes[-1]
        clusters = list(sizes.keys())
        count_vals = list(sizes.values())
        j = 0 if granularity=="topics" else 1;
        ax = axs[i//2][j]
        ax.bar(clusters, count_vals)
        ax.set_title(f"{embedder}; {granularity}")
        
        textstr = \
f"""
Outliers: {outliers:.2f}%
n_neighbors: {n_neighbors}
n_components: {n_components}
min_cluster_size: {min_cluster_size}
min_samples: {min_samples}
""".strip()

        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    subf.suptitle(title)

if __name__ == "__main__":
    df = pd.read_pickle(dataframe_path)

    ten_or_more = df[(df["n_topics"] >= 10) & (df["clustering.min_samples"] > 1)]
    groups = ten_or_more.groupby(["embedder", "granularity"])
    min_outliers = ten_or_more.loc[groups["outliers"].idxmin()].set_index(["embedder", "granularity"])
    min_max_cluster = ten_or_more.loc[groups["max_cluster_size"].idxmin()].set_index(["embedder", "granularity"])

    fig = plt.figure(figsize=(18, 9) ,layout="constrained")
    sub1, sub2 = fig.subfigures(1, 2)
    
    plot_model_results(min_outliers, sub1, "min_outliers")
    plot_model_results(min_max_cluster, sub2, "min_max_cluster")

    # min_max_cluster está dando os melhores resultados
    json_str = min_max_cluster.loc[:, "reduction.n_neighbors":"clustering.min_samples"]\
        .reset_index().to_json(orient="records")
    
    plt.show()