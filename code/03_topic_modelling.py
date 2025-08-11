#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import numpy as np
from umap import UMAP
import chromadb
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import datamapplot
from tqdm import tqdm


def plot(X_2d, y, words_2d, words):
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    
    ax.scatter(X_2d[:,0], X_2d[:,1], c=cmap(y), s=10, marker='o', alpha=0.8)
    ax.axis('off')

    for w in words:
        ax.scatter(words_2d[words==w,0], words_2d[words==w,1], c='orange', marker='o', s=360, alpha=0.2)
        ax.annotate(w, (words_2d[words==w,0], words_2d[words==w,1]))
        ax.axis('off')

    ax.legend()
    plt.show()  

embedders = [
    'sentence-transformers/LaBSE',
    'jinaai/jina-embeddings-v3',
    'intfloat/multilingual-e5-large-instruct',
    'nomic-ai/nomic-embed-text-v2-moe'
]


if __name__ == '__main__':
    for emb_name in (pbar:= tqdm(embedders)):
        pbar.set_postfix_str(emb_name)
    # emb_name = 'jinaai/jina-embeddings-v3' #TODO: change to plot data from a different embedder
        coll_name = emb_name.replace('-', '_').replace(':', '_').replace('/', '_') + '__en'
        client = chromadb.PersistentClient('isidb')
        coll = client.get_collection(coll_name)
        all_data = coll.get(include=['embeddings', 'documents'])

        if not os.path.exists('models/' + coll_name):
            topic_model = BERTopic(nr_topics=30).fit(all_data['documents'], embeddings=all_data['embeddings'])
            topic_model.save('models/' + coll_name, 'safetensors', emb_name, True)
        
        else:
            topic_model = BERTopic.load('models/' + coll_name)
        intplot = topic_model.visualize_document_datamap(all_data['documents'], embeddings=all_data['embeddings'], interactive=True)
        intplot.save('visualizations/' + coll_name + '.html')


    
    # fig = topic_model.visualize_documents(all_data['documents'], embeddings=all_data['embeddings'])
    # p = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine')
    # reduced_embeddings = p.fit_transform(all_data['embeddings'])
    
    # embedder = SentenceTransformer(emb_name, trust_remote_code=True)

    # words = np.array(['crisis', 'famine', 'plague', 'disease', 'sickness', 'death', 'food', 'noble', 'slave', 'cattle', 'dog', 'sheep', 'plough', 'land', 'weather', 'climate', 
    #         'cheese', 'meat', 'fruit', 'milk', 'car', 'road', 'vehicle', 'earth', 'sun', 'moon', 'star', 'arithmetic', 'rain', 'drought',  'flood',
    #         'science', 'harvest', 'wheat', 'rye', 'sorghum', 'bread', 'wine', 'water', 'yeast', 'farm', 'farmer', 'worker',
    #         'king', 'queen', 'servant', 'saint', 'Christ', 'God', 'angel', 'man', 'woman', 'wife', 'war',
    #         'industry', 'battle', 'weapon', 'knife', 'sword', 'steel', 'gold', 'money', 'wealth', 'poverty',
    #         'jewish', 'hebrew', 'abraham', 'muhammad', 'arab', 'muslim', 'african', 'egyptian', 'berber', 'game', 'fight', 'struggle', 'physician'])

    # words_emb = embedder.encode(words, normalize_embeddings=True)
    
    # data.append(all_data['embeddings'])

    # topic_model = BERTopic(nr_topics=30).fit(all_data['documents'], all_data['embeddings'])
    # a = topic_model.get_topic_info()
    # print(a)
    # topics = topic_model.get_document_info(all_data['documents'])['Topic']

    # X = np.concat(data)
    # y = topics.array

    # print(X.shape)

    # for k, v in labels.items():
    #     print(k, v)

    # p = UMAP()
    # X_2d = p.fit_transform(X)
    # words_2d = p.transform(words_emb)

    # np.save('plot_isi_X_2d.npy', X_2d)
    # np.save('plot_isi_words_2d.npy', words_2d)
    # np.save('plot_isi_y.npy', y)
    # np.save('plot_isi_words.npy', words)

    # plot(X_2d, y, words_2d, words)