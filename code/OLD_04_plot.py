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
from sentence_transformers import SentenceTransformer

def plot(X_2d, y, words_2d, words):
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    for c, cl in enumerate(np.unique(y)):
        ax.scatter(X_2d[y==cl,0], X_2d[y==cl,1], c=cmap(c), label=cl, s=10, marker='o', alpha=0.2)
        ax.axis('off')

    for w in words:
        ax.scatter(words_2d[words==w,0], words_2d[words==w,1], c='orange', marker='o', s=360, alpha=0.2)
        ax.annotate(w, (words_2d[words==w,0], words_2d[words==w,1]))
        ax.axis('off')

    ax.legend()
    plt.show()  



if not os.path.exists('plot_isi_X_2d.npy'):
    client = chromadb.PersistentClient('isidb')
    emb_name = 'jinaai/jina-embeddings-v3' #TODO: change to plot data from a different embedder

    embedder = SentenceTransformer(emb_name, trust_remote_code=True)

    words = ['crisis', 'famine', 'plague', 'disease', 'sickness', 'death', 'food', 'noble', 'slave', 'cattle', 'dog', 'sheep', 'plough', 'land', 'weather', 'climate', 
             'cheese', 'meat', 'fruit', 'milk', 'car', 'road', 'vehicle', 'earth', 'sun', 'moon', 'star', 'arithmetic', 'rain', 'drought',  'flood',
             'science', 'harvest', 'wheat', 'rye', 'sorghum', 'bread', 'wine', 'water', 'yeast', 'farm', 'farmer', 'worker',
             'king', 'queen', 'servant', 'saint', 'Christ', 'God', 'angel', 'man', 'woman', 'wife', 'war',
             'industry', 'battle', 'weapon', 'knife', 'sword', 'steel', 'gold', 'money', 'wealth', 'poverty',
             'jewish', 'hebrew', 'abraham', 'muhammad', 'arab', 'muslim', 'african', 'egyptian', 'berber', 'game', 'fight', 'struggle', 'physician']

    words = np.array(words)
    words_emb = embedder.encode(words, normalize_embeddings=True)

    labels = {}
    data = []

    coll_name = emb_name.replace('-', '_').replace(':', '_').replace('/', '_') + '__en'

    coll = client.get_collection(coll_name)
    all_data = coll.get(include=['embeddings', 'metadatas'],
                        where={"sentence": {"$ne" : "all"}}
    )
    labels['en'] = all_data['embeddings'].shape[0]
    data.append(all_data['embeddings'])

    X = np.concat(data)
    y = np.concat([np.repeat(k, v) for k, v in labels.items()])

    for k, v in labels.items():
        print(k, v)

    p = UMAP()
    X_2d = p.fit_transform(X)
    words_2d = p.transform(words_emb)

    np.save('plot_isi_X_2d.npy', X_2d)
    np.save('plot_isi_words_2d.npy', words_2d)
    np.save('plot_isi_y.npy', y)
    np.save('plot_isi_words.npy', words)

    # plot(X_2d, y, words_2d, words)
else:
    X_2d = np.load('plot_isi_X_2d.npy')
    words_2d = np.load('plot_isi_words_2d.npy')
    y = np.load('plot_isi_y.npy')
    words = np.load('plot_isi_words.npy')
    print(np.unique_counts(y))

    # plot(X_2d, y, words_2d, words)