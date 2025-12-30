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
from sys import argv, stderr

def get_repres_docs(topic:int, doc_info:pd.DataFrame) -> pd.DataFrame:
    return doc_info[(doc_info["Topic"] == topic) & (doc_info["Representative_document"])]

def plot_books_per_topic(doc_info: pd.DataFrame):
    topic_list = np.sort(doc_info["Topic"].unique())
    book_list = np.sort(doc_info["book"].unique())
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"]
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
    

if __name__ == "__main__":

    all_data = {
        granularity: getDocs("nomic_ai_nomic_embed_text_v2_moe__en", granularity, include=["documents", "metadatas"])
        for granularity in getArg("granularity")
    }

    # Get topic model configurations
    with open(selected_confs_json_path, 'r') as json_file:
        confs = json.load(json_file)

    coll_name = 'jinaai_jina_embeddings_v3__en_2'
    # granularity = 'sentences'

    if len(argv) < 3:
        print("Error", file=stderr)
        exit(1)

    granularity=argv[1]
    if granularity not in ["sentences", "topics"]:
        print("Wrong args!", file=stderr)
        exit(1)

    output_dir = argv[2]

    n_reps = 5 if granularity == "sentences" else 3

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
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"]
    book_titles.index+=1
    doc_info["book_title"] = doc_info["book"].map(book_titles)
    # book_dist = plot_books_per_topic(doc_info)
    # topics_per_book = plot_topics_per_book(doc_info, topic_info)
    topic_info.drop(-1, inplace=True)
    book_dist = plot_books_per_topic(doc_info)
    # wcs = [ create_wordcloud(topic_model, topic) for topic in topic_info.index ]

    for (topic_n, topic_name), topic_df in doc_info[(doc_info["Topic"] >=0) & (doc_info["Topic"] < 6)].groupby(["Topic", "Name"]):
        with open(f"{output_dir}/topics_{granularity}_{topic_n + 1}.tex", 'w') as output_file:
            representation = ", ".join(topic_df.iloc[0]["Representation"][:10])
            print(f"\\subsubsection{{Tópico {topic_n + 1}: \\textit{{{representation}}}}}", file=output_file)
            print(f"\\label{{tema:{granularity}_{topic_n + 1}}}", file=output_file)
            print("\\begin{description}", file=output_file)
            print("\\footnotesize", file=output_file)
            most_rep_books = book_dist[topic_n].sort_values(ascending=False)[:5]
            # most_rep_books = most_rep_books[most_rep_books >= (.1 if granularity == "topics" else .15)]
            topic_df = topic_df[topic_df["book_title"].isin(most_rep_books.index)]
            topic_df["book_val"] = topic_df["book_title"].map(most_rep_books)
            topic_df = topic_df.sort_values(by="book_val", ascending=False)
            # topic_df = topic_df[topic_df["Representative_document"]]
            for (book_n, book), book_df in topic_df.groupby(["book", "book_title"], sort=False):
                bval = book_df["book_val"].iloc[0]
                print(f"\\item [Livro: {book_n} - {book} ({100 * bval:.2f}\%)]\\hfill", file=output_file)
                print("\\begin{itemize}", file=output_file)
                for chapter, doc, section, sentence, edition in book_df.groupby("chapter")[["Document", "section", "sentence", "edition"]].first()[:n_reps].itertuples():
                    # if granularity == "topics": sentence = "-"
                    print(f"\\item[\\textbf{{isi\_{edition}.{book_n}.{chapter}.{section}.S{sentence}}}]", file=output_file)
                    print(f"\\textit{{{doc}}}", file=output_file)

                print("\\end{itemize}", file=output_file)
                
            print("\\end{description}", file=output_file)