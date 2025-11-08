#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)

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

    for conf in iterConfigs(["embedder"]):
        # granularity = conf["granularity"]
        granularity = "topics"
        emb_name = conf["embedder"]
        coll_name = getCollName(emb_name)
        data = all_data[granularity]
        docs = data["documents"]
        assert docs is not None
        X = pd.DataFrame(np.load(f"./reduced_embds/{granularity}_{coll_name}_2d.npy"), columns=["X", "Y"])
        n_docs = X.shape[0]

        for conf2 in iterConfigs(["reduction", "clustering", "n_topics"]):
            n_topics = conf2["n_topics"]
            topic_model = BERTopic.load(getModelFilePath({**conf, **conf2, "granularity":granularity}))
            doc_info = topic_model.get_document_info(docs)
            doc_info = pd.concat([doc_info, X], axis=1)
            topic_info = topic_model.get_topic_info().set_index("Topic")
            # topics = topic_model.topics_
            fig, _ = plot(doc_info, topic_info, f"Topicos por documento", f"Embedder: {emb_name}; granularidade: {granularity}; Tópicos: {n_topics}")
            fig.savefig(f"visualizations/images/{granularity}_{coll_name}_{n_topics}.png")
            plt.close()
            plt.clf()
            plt.cla()
            
            svg = pd.Series([ create_wordcloud(topic_model, topic) for topic in topic_info.index ], name="svg", index=topic_info.index)
            topic_info["svg"] = svg

            outlier_count: int = topic_info.loc[-1, "Count"].item() if -1 in topic_info.index else 0
            outlier_pct = 100 * outlier_count / n_docs
            topic_count = topic_info["Name"].count().item()
            doc_datamap = topic_model.visualize_document_datamap(docs, reduced_embeddings=X.to_numpy(), interactive=True)
            # h_topics = model.hierarchical_topics(docs)
            # h_docs = model.visualize_hierarchical_documents(docs, h_topics, reduced_embeddings=reduced_embds, hide_document_hover=False)
            # h_docs.write_html('v.html', include_plotlyjs='cdn')


            data = {
                'nome_embedder': coll_name,
                'dmapplt': escape(str(doc_datamap)),
                'n_topics': n_topics,
                # 'hierarchical': h_docs.to_html(full_html=False, include_plotlyjs='cdn'),
                'nivel': granularity,
                'outlier_pct': outlier_pct,
                'outlier_count': outlier_count,
                'topic_count': topic_count,
                'topics': topic_info,
                **conf2["clustering"],
                **conf2["reduction"]
            }

            with open(f"./visualizations/html/{granularity}_{coll_name}_{n_topics}.html", 'w') as vis_file:
                print(template.render(data), file=vis_file)

    # for name, gran_dict in topic_models.items():
    #     for granularity, model in gran_dict.items():

            