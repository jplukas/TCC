#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)

createTopicModel = __import__("03_topic_modelling", fromlist="createTopicModel").createTopicModel

import os
import numpy as np
from umap import UMAP
import chromadb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
from utils import *
from bertopic import BERTopic
from jinja2 import Environment, FileSystemLoader
import seaborn as sns
import pandas as pd
from html import escape
from wordcloud import WordCloud
import json
import datamapplot
import plotly.express as px

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

    bpt = {
            topic.item(): {
                book_titles[book.item()]:
                len(
                    doc_info[(doc_info["book"] == book) & (doc_info["Topic"] == topic)]
                ) / (docs_per_book[book.item()] if topic != -1 else non_noise_docs_per_book[book.item()])
                for book in book_list
            }
        for topic in topic_list
    }
    
    return {
        topic: px.bar(x=counts.keys(), y=counts.values()).to_html(full_html=False, include_plotlyjs='cdn')
        for topic, counts in bpt.items() 
    }

def plot_topics_per_book(doc_info: pd.DataFrame, topic_info:pd.DataFrame):
    topic_list = topic_info.index
    topic_titles = topic_info["Name"]
    book_list = np.sort(doc_info["book"].unique())
    book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"]
    book_titles.index+=1
    docs_per_book = {k.item(): len(v) for k, v in doc_info.groupby("book").indices.items()}
    tpb = {
            book_titles[book]: pd.Series({
                topic_titles[topic]:
                len(
                    doc_info[(doc_info["book"] == book) & (doc_info["Topic"] == topic)]
                ) / docs_per_book[book]
                for topic in topic_list
            }).sort_values(ascending=False).drop(labels=topic_titles[-1]).where(lambda item: item > 0.01).dropna()
        for book in book_list
    }
    # non_noise_docs_per_book = {
    #     k.item(): len(v) for k, v in
    #     doc_info[doc_info["Topic"] != -1].groupby("book").indices.items()
    # }
    
    return {
        book: px.bar(x=counts.index, y=counts.values).to_html(full_html=False, include_plotlyjs='cdn')
        for book, counts in tpb.items() 
    }

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
    wc = wc.generate_from_frequencies(text)
    return wc.to_svg(embed_font=True, optimize_embedded_font=False)

def plot(doc_info: pd.DataFrame, topics:pd.DataFrame, title: str, subtitle: str | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    n_clusters = topics.shape[0]

    if(-1 in topics.index):
        n_clusters -= 1
        
    colors = sns.color_palette("tab10", n_colors=n_clusters)
    custom_palette = {k: colors[i] for i, k in enumerate(topic_info.loc[0:, "Name"])}

    if(-1 in topics.index):
        noise = topics["Name"][-1]
        custom_palette[noise] = (.1, .1, .1)

    sns.scatterplot(data=doc_info, x="X", y="Y", hue="Name", alpha=0.3, ax=ax, palette=custom_palette)
    fig.suptitle(title)
    if subtitle: ax.set_title(subtitle)
    ax.axis('off')
    # ax.legend()
    return fig, ax

if __name__ == "__main__":
    granularities = getArg("granularity")

    all_data = {
        granularity: getDocs("nomic_ai_nomic_embed_text_v2_moe__en", granularity, include=["documents", "metadatas"])
        for granularity in granularities
    }

    env = Environment(loader=FileSystemLoader("templates/"))
    template = env.get_template("template.html.jinja")

    # Get topic model configurations
    with open(selected_confs_json_path, 'r') as json_file:
        confs = json.load(json_file)

    for conf in confs:
        print(conf)
        reduction = {
            "n_neighbors":conf["reduction.n_neighbors"],
            "n_components":conf["reduction.n_components"],
            "densmap": False
        }
        clustering = {
            "min_cluster_size":conf["clustering.min_cluster_size"],
            "min_samples":conf["clustering.min_samples"],
        }

        granularity = conf["granularity"]
        embedder = conf["embedder"]
        n_topics = 20

        conf = {
            "reduction": reduction,
            "clustering": clustering,
            "embedder": embedder,
            "granularity": granularity,
            "n_topics": n_topics,
        }

        coll_name = getCollName(embedder)
        data = all_data[granularity]
        doc_metadata = pd.DataFrame(data["metadatas"]).rename({"topic":"section"}, axis=1)
        doc_metadata["book"] = doc_metadata["book"].astype(int)
        docs = data["documents"]
        assert docs is not None

        reduction_2d = {**reduction, "n_components": 2}
        reduction = np.load(getReductionFilePath(conf))
        X_2d_filepath = getReductionFilePath({**conf, "reduction": reduction_2d})
        X = pd.DataFrame(np.load(X_2d_filepath), columns=["X", "Y"])
        n_docs = X.shape[0]
        
        topic_model = createTopicModel(docs, embeddings=reduction, **conf)
        # ps = (1 - topic_model.hdbscan_model.probabilities_)
        # ps *= (topic_model.hdbscan_model._outlier_scores or 1)
        # topic_model = BERTopic.load(getModelFilePath({**conf}))
        # topic_model.reduce_topics(docs, nr_topics=20, use_ctfidf=True)
        doc_info = topic_model.get_document_info(docs)
        # doc_info["Ps"] = ps
        doc_info["Topic"] = doc_info["Topic"].astype(int)
        doc_info = pd.concat([doc_info, X, doc_metadata], axis=1)
        topic_info = topic_model.get_topic_info().set_index("Topic")
        book_titles = pd.DataFrame(getOpt("book_struct"))["title_en"]
        book_titles.index+=1
        doc_info["book_title"] = doc_info["book"].map(book_titles)
        topic_info["book_dist"] = plot_books_per_topic(doc_info)
        topics_per_book = plot_topics_per_book(doc_info, topic_info)
        svg = pd.Series([ create_wordcloud(topic_model, topic) for topic in topic_info.index ], name="svg", index=topic_info.index)
        topic_info["svg"] = svg

        outlier_count: int = topic_info.loc[-1, "Count"].item() if -1 in topic_info.index else 0
        outlier_pct = 100 * outlier_count / n_docs
        topic_count = topic_info.index.max().item() + 1
        h_topics = topic_model.hierarchical_topics(docs)
        hierarchy = get_hierarchies(topic_info["Name"], h_topics)
        labels = [doc_info["Topic"].map(h) for h in hierarchy]
        doc_datamap = datamapplot.create_interactive_plot(
            X.to_numpy(), *labels, hover_text=docs,
            noise_label=topic_info.loc[-1, "Name"],
            colormaps={
                "Books": doc_info["book_title"]
            }
        )

        topic_distribution = px.bar(topic_info, x="Name", y="Count").to_html(full_html=False, include_plotlyjs='cdn')
        # rep_docs = {topic: get_repres_docs(topic, doc_info) for topic in topic_info.index}
        rep_docs = {topic: get_repres_docs(topic, doc_info) for topic in topic_info.index}

        topic_info["Representative_Docs"] = rep_docs

        data = {
            'nome_embedder': coll_name,
            'topics_per_book': topics_per_book,
            'dmapplt': escape(str(doc_datamap)),
            'n_topics': n_topics,
            'nivel': granularity,
            'outlier_pct': outlier_pct,
            'outlier_count': outlier_count,
            'topic_count': topic_count,
            'topics': topic_info,
            'distribution': topic_distribution,
            **conf["clustering"],
            **conf["reduction"]
        }

        HTMLFilePath = getHTMLFilePath(conf)
        with open(HTMLFilePath, 'w') as vis_file:
            print(template.render(data), file=vis_file)

            
        # topics = topic_model.topics_
        # fig, _ = plot(doc_info, topic_info, f"Topicos por documento", f"Embedder: {embedder}; granularidade: {granularity}; Tópicos: {n_topics}")
        # fig.savefig(f"visualizations/images/{granularity}_{coll_name}_{n_topics}.png")
        # plt.close()
        # plt.clf()
        # plt.cla()
        
        # barchart = topic_model.visualize_barchart(top_n_topics=20).to_html(full_html=False, include_plotlyjs='cdn')
        
        # h_docs = topic_model.visualize_hierarchy(hierarchical_topics=h_topics).to_html(full_html=False, include_plotlyjs='cdn')

        # h_docs.write_html('v.html', include_plotlyjs='cdn')