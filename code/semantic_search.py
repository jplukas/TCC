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
# from sklearn.metrics.pairwise import cosine_similarity

# def plot(X_2d, names, words_2d, words):
#     texts=[]
#     cmap = plt.get_cmap('tab10')

#     fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))

#     for c, cl in enumerate(np.unique(names)):
#         ax.scatter(X_2d[names==cl,1], X_2d[names==cl,0], label=cl, s=20, marker='o', alpha=0.2)
#         ax.axis('off')

#     for w in words:
#         ax.scatter(words_2d[words==w,1], words_2d[words==w,0], c='orange', marker='o', s=360, alpha=0.2)
#         texts.append(ax.text(words_2d[words==w,1], words_2d[words==w,0], w))
#         # ax.annotate(w, (words_2d[words==w,0], words_2d[words==w,1]))
#         ax.axis('off')

#     leg = ax.legend(title="Livros")
#     adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
#     for handle in leg.legend_handles:
#         handle.set_alpha(1.0)
#     plt.tight_layout()
#     plt.show()  

# def formatEllipsis(s:str, max:int = 25)-> str:
#     if len(s) <= max:
#         return s
#     return s[:max - 3] + "..."

def cosine_similarity_numpy(vec_a, vec_b):
    """
    Calculates the cosine similarity between two 1-D numpy arrays.
    """
    # Ensure inputs are numpy arrays
    A = np.array(vec_a)
    B = np.array(vec_b)
    
    # Calculate the dot product
    dot_product = np.dot(A, B)
    
    # Calculate the L2 norms (magnitudes) of both vectors
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    
    # Handle division by zero for zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    # Calculate cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

if __name__ == "__main__":
    # granularities = getArg("granularity")
    # embedders = getArg("embedder")
    # book_titles = {str(i + 1): formatEllipsis(d["title_en"]) for i, d in enumerate(getOpt("book_struct"))}

    embedders = ["jinaai/jina-embeddings-v3"]

    # all_data = {
    #     embedder: {granularity: getDocs(getCollName(embedder), granularity, include=["documents", "metadatas", "embeddings"])
    #     for granularity in granularities} for embedder in embedders
    # }

    varrao = [
        "The beard is called arista from the fact that it is the first part to dry (arescere).",
        "Amurca, which is a watery fluid, after it is pressed from the olives is stored along with the dregs in an earthenware vessel."
    ]

    # client = getClient()
    # collections = {"jinaai/jina-embeddings-v3":client.get_collection(name="jinaai_jina_embeddings_v3__en_2")}

    collections = getCollections(embedders)
    book_titles = {str(i + 1): d["title_en"] for i, d in enumerate(getOpt("book_struct"))}

    for emb_name, collection in collections.items():
        print(f"Embedder:{emb_name}")
        device = "cuda" if emb_name not in [
            'intfloat/multilingual-e5-large-instruct',
            # 'nomic-ai/nomic-embed-text-v2-moe',
        ] else "cpu"
        embedder_model = SentenceTransformer(emb_name, trust_remote_code=True, device="cuda")
        for query in varrao:
            print("Query:")
            print(f"\"{query}\"\n")
            query_embedding = embedder_model.encode(query, normalize_embeddings=True,
                                                    # task="separation", prompt_name="separation"
            )
            r = collection.query(query_embedding, n_results=3, include=["documents", "metadatas", "distances", "embeddings"], where={"sentence": {"$ne" : "all"}})
            res = {k: v[0] for k, v in r.items() if k in ["ids", "documents", "metadatas", "distances", "embeddings"]}
            print("Most similar passages:")
            for id, embeddings, passage, metadata, distance in zip(*res.values()):

                edition = metadata["edition"]
                section = metadata["topic"]
                sentence = metadata["sentence"]
                chapter = metadata["chapter"]
                book = metadata["book"]
                book_title = book_titles[book]

                # print(id)
                print(f"Livro: {book_title}; capítulo: {chapter}; seção: {section}; sentença: {sentence}")
                print(f"\"{passage}\"")
                # print(distance)
                print(f"Similaridade de cosseno: {cosine_similarity_numpy(embeddings, query_embedding):.5f}")
                print("-------------------------------------------\n")
        
        


    # for embedder in embedders:
        # embedder_model = SentenceTransformer(embedder, trust_remote_code=True)
        # word_embs = embedder_model.encode(varrao, normalize_embeddings=True)
        
        # for granularity in granularities:

            # embeddings = all_data[embedder][granularity]["embeddings"]
            # docs = all_data[embedder][granularity]["documents"]
            # meta = all_data[embedder][granularity]["metadatas"]
            # umap = UMAP(n_neighbors=5, n_components=2, metric='cosine', min_dist=0, random_state=42)
            # X_2d = umap.fit_transform(embeddings)
            # words_2d = umap.transform(word_embs)
            # book_names = np.array([book_titles[d["book"]] for d in meta])
            
            # plot(X_2d, book_names, words_2d, words)
        