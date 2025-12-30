#!/usr/bin/env python3
# -*- coding: utf-8 -*-

selected_topics = []

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)

createTopicModel = __import__("03_topic_modelling", fromlist="createTopicModel").createTopicModel

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from utils import *
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import json
import plotly.express as px
from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.ticker as mticker

N_ROWS = 6
IMAGE_WIDTH_INCHES = 2.0 # Define the fixed width of the image column in inches
COUNT_PLOT_WIDTH_SCALED = 1.0 # This column can scale

def get_repres_docs(topic:int, doc_info:pd.DataFrame) -> pd.DataFrame:
    return doc_info[(doc_info["Topic"] == topic) & (doc_info["Representative_document"])]

def plot_books_per_topic(doc_info: pd.DataFrame):
    topic_list = np.sort(doc_info["Topic"].unique())
    book_list = np.sort(doc_info["book"].unique())
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"].apply(formatEllipsis)
    book_titles.index+=1
    docs_per_book = {k.item(): len(v) for k, v in doc_info.groupby("book").indices.items()}
    non_noise_docs_per_book = {
        k.item(): len(v) for k, v in
        doc_info[doc_info["Topic"] != -1].groupby("book").indices.items()
    }

    return {
            topic.item(): pd.Series({
                book_titles[book.item()]:
                len(
                    doc_info[(doc_info["book"] == book) & (doc_info["Topic"] == topic)]
                ) / (docs_per_book[book.item()] if topic != -1 else non_noise_docs_per_book[book.item()])
                for book in book_list
            })
        for topic in topic_list
    }

def plot_topics_per_book(doc_info: pd.DataFrame, topic_info:pd.DataFrame):
    topic_list = topic_info.index
    topic_titles = topic_info["Name"]
    book_list = np.sort(doc_info["book"].unique())
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"]
    book_titles = book_titles.apply(formatEllipsis)
    book_titles.index+=1
    docs_per_book = {k.item(): len(v) for k, v in doc_info.groupby("book").indices.items()}
    return {
            book_titles[book]: pd.Series({
                topic_titles[topic]:
                len(
                    doc_info[(doc_info["book"] == book) & (doc_info["Topic"] == topic)]
                ) / docs_per_book[book]
                for topic in topic_list
            }).drop(labels=topic_titles[-1])
        for book in book_list
    }

def formatEllipsis(s:str, max:int = 25)-> str:
    if len(s) <= max:
        return s
    return s[:max - 3] + "..."

def get_hierarchies(topics: dict, h_topics):
    ts:dict = topics
    r = [topics]
    for level in h_topics.sort_values("Distance").itertuples():
        topics_to_merge = level.Topics
        l = ts.copy()
        for t in topics_to_merge:
            l[t] = level.Parent_Name
        ts = l
        r.append(ts)
    return r


def save_wordcloud(model, topic, filename):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=100)
    return wc.generate_from_frequencies(text)
    # return wc.to_svg(embed_font=True, optimize_embedded_font=False)

def func(x, prefixes):
    masks = {prefix: x.index.str.startswith(prefix) for prefix in prefixes}
    not_selected = ~np.logical_or(*masks.values())
    t = {prefix: {k.removeprefix(prefix + "_"): v for k, v in x[mask].items()} for prefix, mask in masks.items()}
    return pd.Series({**x[not_selected], **t})

# def plot(doc_info: pd.DataFrame, topics:pd.DataFrame, title: str, subtitle: str | None = None):
#     fig, ax = plt.subplots(1, 1, figsize=(20, 20))
#     n_clusters = topics.shape[0]

#     if(-1 in topics.index):
#         n_clusters -= 1
        
#     colors = sns.color_palette("tab10", n_colors=n_clusters)
#     custom_palette = {k: colors[i] for i, k in enumerate(topic_info.loc[0:, "Name"])}

#     if(-1 in topics.index):
#         noise = topics["Name"][-1]
#         custom_palette[noise] = (.1, .1, .1)

#     sns.scatterplot(data=doc_info, x="X", y="Y", hue="Name", alpha=0.3, ax=ax, palette=custom_palette)
#     fig.suptitle(title)
#     if subtitle: ax.set_title(subtitle)
#     ax.axis('off')
#     # ax.legend()
#     return fig, ax

if __name__ == "__main__":

    all_data = {
        granularity: getDocs("nomic_ai_nomic_embed_text_v2_moe__en", granularity, include=["documents", "metadatas"])
        for granularity in getArg("granularity")
    }

    # Get topic model configurations
    with open(selected_confs_json_path, 'r') as json_file:
        confs = json.load(json_file)

    coll_name = 'jinaai_jina_embeddings_v3__en_2'
    granularity = 'sentences'

    df = pd.DataFrame(confs).agg(func, "columns", ["reduction", "clustering"])
    config = df[(df["coll_name"] == coll_name) & (df["granularity"] == granularity)].iloc[0].to_dict()

    data = all_data[granularity]
    docs = data["documents"]
    embeddings = np.load(getReductionFilePath(config))
    topic_model = createTopicModel(docs, embeddings=embeddings, **config)
    doc_info = topic_model.get_document_info(docs)
    doc_metadata = pd.DataFrame(data["metadatas"]).rename({"topic":"section"}, axis=1)
    doc_metadata["book"] = doc_metadata["book"].astype(int)
    doc_info = pd.concat([doc_info, doc_metadata], axis=1)
    doc_info["Topic"] = doc_info["Topic"].astype(int)
    doc_info = doc_info[doc_info["Topic"] < 20]
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[(topic_info["Topic"] < 20)].set_index("Topic")
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"].apply(formatEllipsis)
    book_titles.index+=1
    doc_info["book_title"] = doc_info["book"].map(book_titles)
    book_dist = plot_books_per_topic(doc_info)
    topics_per_book = plot_topics_per_book(doc_info, topic_info)
    topic_info.drop(-1, inplace=True)
    wcs = [ create_wordcloud(topic_model, topic) for topic in topic_info.index ]
    
    
    N_ROWS = 6
    IMAGE_WIDTH_INCHES = 2.5 # Define the fixed width of the image column in inches
    COUNT_PLOT_WIDTH_SCALED = 1.0 # This column can scale

    fig = plt.figure(figsize=(10, N_ROWS * 2))

    # --- New: Store references to the first axis created in each column ---
    # This will be the master axis that others share
    master_ax_count = None
    master_ax_img = None
    # --- End New ---

    gs = plt.GridSpec(N_ROWS, 1, top=.99, bottom=.2, left=0, right=1, hspace=0)
    # gs = plt.GridSpec(N_ROWS, 1, hspace=0)
    h = [Size.Fixed(0.1), Size.Scaled(COUNT_PLOT_WIDTH_SCALED), Size.Fixed(-0.15), 
        Size.Fixed(IMAGE_WIDTH_INCHES), Size.Fixed(-0.2)]
    v = [Size.Fixed(0.2), Size.Scaled(1), Size.Fixed(0.0)]

    for i in range(N_ROWS):
        counts = book_dist[i]
        gs_row = gs[i]
        host = fig.add_subplot(gs_row)
        host.axis('off')
        divider = Divider(fig, host.get_position(), h, v, aspect=False)
        
        # Left column: Count Plot (Scaled)
        ax_count = fig.add_axes(
            divider.get_position(), 
            axes_locator=divider.new_locator(nx=1, ny=1),
            # --- New: Share X-axis if a master exists ---
            sharex=master_ax_count
            # --- End New ---
        )
        
        sns.barplot(x=counts.index, y=counts.values, ax=ax_count)
        # sns.countplot(x=np.random.choice(['A', 'B', 'C', 'D'], 20), ax=ax_count)
        if i < N_ROWS - 1:
            ax_count.xaxis.set_visible(False)
        else:
            ax_count.tick_params('x', labelrotation=90)
        # ax_count.axis('off')
        
        # Right column: Image (Fixed Width)
        ax_img = fig.add_axes(
            divider.get_position(), 
            axes_locator=divider.new_locator(nx=3, ny=1),
            # --- New: Share X-axis if a master exists ---
            sharex=master_ax_img
            # --- End New ---
        )
        ax_img.imshow(wcs[i], interpolation="bilinear")
        # ax_img.imshow(np.random.rand(50, 50), cmap='gray')
        ax_img.axis('off')

        # --- New: Assign the current axes as the master for the next iteration if they don't exist ---
        if master_ax_count is None:
            master_ax_count = ax_count
        if master_ax_img is None:
            master_ax_img = ax_img
        # --- End New ---

    fig.set_constrained_layout(True) 
    plt.show()
