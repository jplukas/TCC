#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)


import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
from utils import *
from adjustText import adjust_text

def plot(X_2d, names, words_2d, words):
    texts=[]
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))

    for c, cl in enumerate(np.unique(names)):
        ax.scatter(X_2d[names==cl,1], X_2d[names==cl,0], label=cl, s=20, marker='o', alpha=0.2)
        ax.axis('off')

    for w in words:
        ax.scatter(words_2d[words==w,1], words_2d[words==w,0], c='orange', marker='o', s=360, alpha=0.2)
        texts.append(ax.text(words_2d[words==w,1], words_2d[words==w,0], w))
        # ax.annotate(w, (words_2d[words==w,0], words_2d[words==w,1]))
        ax.axis('off')

    leg = ax.legend(title="Livros")
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    plt.tight_layout()
    plt.show()  

def formatEllipsis(s:str, max:int = 25)-> str:
    if len(s) <= max:
        return s
    return s[:max - 3] + "..."

if __name__ == "__main__":
    # granularities = getArg("granularity")
    # embedders = getArg("embedder")
    book_titles = {str(i + 1): formatEllipsis(d["title_en"]) for i, d in enumerate(getOpt("book_struct"))}

    granularities = ["topics"]
    embedders = ["jinaai/jina-embeddings-v3"]

    all_data = {
        embedder: {granularity: getDocs(getCollName(embedder), granularity, include=["documents", "metadatas", "embeddings"])
        for granularity in granularities} for embedder in embedders
    }
    words = np.array(getOpt("words"))

    for embedder in embedders:
        embedder_model = SentenceTransformer(embedder, trust_remote_code=True)
        word_embs = embedder_model.encode(words, normalize_embeddings=True)
        for granularity in granularities:
            embeddings = all_data[embedder][granularity]["embeddings"]
            docs = all_data[embedder][granularity]["documents"]
            meta = all_data[embedder][granularity]["metadatas"]
            umap = UMAP(n_neighbors=5, n_components=2, metric='cosine', min_dist=0, random_state=42)
            X_2d = umap.fit_transform(embeddings)
            words_2d = umap.transform(word_embs)
            book_names = np.array([book_titles[d["book"]] for d in meta])
            
            plot(X_2d, book_names, words_2d, words)
        