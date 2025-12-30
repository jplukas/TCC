#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from utils import *

bars = None

def plot_model_results(m, fig, title):

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs = fig.subplots(len(m["sizes"]) // 2, 2)

    for i, row in enumerate(m.itertuples()):

        (coll_name, granularity), n_topics, outliers, max_cluster_size, n_neighbors, n_components, min_dist, low_memory, metric, random_state, densmap, min_cluster_size, min_samples, cluster_selection_method, sizes = row
        if -1 in sizes: del sizes[-1]
        clusters = list(sizes.keys())
        n_clusters = len(clusters)
        count_vals = list(sizes.values())
        j = 0 if granularity=="topics" else 1;
        ax = axs[i//2][j]
        ax.bar(clusters, count_vals)
        ax.set_title(f"{coll_name}; {granularity}")

        textstr = \
f"""
n_clusters {n_clusters}
outliers: {outliers:.2f}%
n_neighbors: {n_neighbors}
n_components: {n_components}
min_cluster_size: {min_cluster_size}
min_samples: {min_samples}
""".strip()

        ax.text(0.6, 0.95, textstr, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    fig.suptitle(title)

if __name__ == "__main__":
    df = pd.read_pickle(dataframe_path)

    ten_or_more = df[(df["n_topics"] >= 10) & (df["clustering_min_samples"] > 1)]
    groups = ten_or_more.groupby(["coll_name", "granularity"])
    min_outliers = ten_or_more.loc[groups["outliers"].idxmin()].set_index(["coll_name", "granularity"])
    min_max_cluster = ten_or_more.loc[groups["max_cluster_size"].idxmin()].set_index(["coll_name", "granularity"])

    
    fig1 = plt.figure(figsize=(12, 9) ,layout="constrained")
    plot_model_results(min_outliers, fig1, "min_outliers")
    plt.show()
    # fig1.savefig("min_outliers")

    # fig2 = plt.figure(figsize=(12, 9) ,layout="constrained")
    # plot_model_results(min_max_cluster, fig2, "min_max_cluster")
    # fig2.savefig("min_max_cluster")

    # min_max_cluster está dando os melhores resultados
    # json_str = min_max_cluster.loc[:, "reduction_n_neighbors":"clustering_min_samples"]\
    #     .reset_index().to_json(orient="records")
    
    # with open(selected_confs_json_path, 'w') as json_file:
    #     json_file.write(json_str)