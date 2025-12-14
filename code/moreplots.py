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
    # with open(selected_confs_json_path, 'r') as json_file:
    #     confs = json.load(json_file)

    embedder = 'jinaai/jina-embeddings-v3'
    granularity = 'topics'

    conf = {'embedder': embedder, 'granularity': granularity, 'reduction.n_neighbors': 5, 'reduction.n_components': 20, 'clustering.min_cluster_size': 35, 'clustering.min_samples': 5}
    reduction = {
        "n_neighbors":conf["reduction.n_neighbors"],
        "n_components":conf["reduction.n_components"],
        "densmap": False
    }
    clustering = {
        "min_cluster_size":conf["clustering.min_cluster_size"],
        "min_samples":conf["clustering.min_samples"],
    }
    n_topics = 20

    config = {
        "reduction": reduction,
        "clustering": clustering,
        "embedder": embedder,
        "granularity": granularity,
        "n_topics": n_topics,
    }
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
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"]
    book_titles.index+=1
    doc_info["book_title"] = doc_info["book"].map(book_titles)
    book_dist = plot_books_per_topic(doc_info)
    topics_per_book = plot_topics_per_book(doc_info, topic_info)
    topic_info.drop(-1, inplace=True)
    wcs = [ create_wordcloud(topic_model, topic) for topic in topic_info.index ]

    for topic_name, topic_df in doc_info[doc_info["Topic"] >=0 ].groupby("Name"):
        print(f"Topic: {topic_name}")
        for (book, chapter), doc, section in topic_df.groupby(["book_title", "chapter"])[["Document", "section"]].first().itertuples():
            print(f"book: {book}, chapter {chapter}, section {section}")
            print(f"\"{doc}\"")
        print("**********************************************************")
        print()

    
    # fig, ax = plt.subplots(6, 2, sharex="col")

    # for i, (book, counts) in enumerate(topics_per_book.items()):
    #     if i >=10: break
    #     sns.barplot(x=counts.index, y=counts.values, ax=ax[i])