#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)

import os
import numpy as np
from hdbscan import HDBSCAN, flat

from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from typing import Any, List
from utils import *
import pandas as pd


def createTopicModel(docs: List[Any], # embedder: str, coll_name: str, granularity: granType,
    clustering: dict[str, Any],
    embeddings: Any, reduction:dict[str, Any] | None = None, topic_reduction_strategy: str | None = None, n_topics: int = None,
    outlier_reduction_strategy: str | None = None, embedder: Any | None = None,
    granularity: Any | None = None, **kwargs):

    from bertopic import BERTopic
    
    hdbscan_model = HDBSCAN(core_dist_n_jobs=-1, prediction_data=True, **clustering)
    # hdbscan_model = flat.HDBSCAN_flat(X=embeddings, core_dist_n_jobs=-1, prediction_data=True, n_clusters=n_topics, **clustering)
    
    vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    ctfidf = ClassTfidfTransformer(True, True)

    return BERTopic(
                top_n_words=100,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf,
                umap_model=BaseDimensionalityReduction(),
                nr_topics=None,
                calculate_probabilities=True,
            ).fit(docs, embeddings)

if __name__ == "__main__":

    #if not os.path.exists(dataframe_path):
    if True:
        from bertopic import BERTopic

        all_data = {
            granularity: getDocs("nomic_ai_nomic_embed_text_v2_moe__en", granularity, include=["documents", "metadatas"])
            for granularity in getArg("granularity")
        }
            
        # chosen_embedder = "sentence-transformers/LaBSE"
        # chosen_granularity = "topics"

        res = []
        size_vec = []
        
        for c in iterConfigs(["granularity", "coll_name"]):
            granularity = c["granularity"]
            
            #embedder = c["embedder"]

            # if granularity != chosen_granularity or embedder != chosen_embedder: continue
            
            docs = all_data[granularity]["documents"]
            n_docs = len(docs)
            for r in iterConfigs(["reduction"]):
                cc = {**c, **r}
                n_components = r["reduction"]["n_components"]
                if n_components < 5: continue
                reduction_filepath = getReductionFilePath(cc)
                embeddings=np.load(reduction_filepath)
                # print("Computing pairwise distances...", end="")
                # precomputed_distances = pairwise_distances(embeddings, n_jobs=1).astype(np.float64)
                # print(" done!")

                for conf in iterConfigs(["clustering", "n_topics"]):
                        config = {**cc, **conf}
                        model_filepath = getModelFilePath(config)
                        print(model_filepath)
                        #model_filepath = "naoexiste"
                        if not os.path.exists(model_filepath):
                            min_samples = config["clustering"]["min_samples"]
                            min_cluster_size = config["clustering"]["min_cluster_size"]
                            if min_cluster_size < min_samples: continue
                            
                            topic_model = createTopicModel(
                                docs, embeddings=embeddings, **config
                            )
                            topic_model.save(model_filepath, serialization='safetensors', save_ctfidf=True)
                        else:
                            topic_model = BERTopic.load(model_filepath)
                            
                        sizes = topic_model.topic_sizes_
                        n_topics = max(sizes.keys()) + 1
                        print(config)
                        print(f"Number of topics found: {n_topics}")
                        max_cluster_size = max(sizes.values())
                        outliers = 100 * sizes[-1] / n_docs if -1 in sizes else 0
                        print(f"Outliers: {outliers:.2f}%")
                        
                        res.append({
                            **config,
                            "n_topics": n_topics, 
                            "outliers": outliers,
                            "max_cluster_size": max_cluster_size,
                        })
                        size_vec.append(sizes)

        df = pd.json_normalize(res, sep="_")
        df["sizes"] = size_vec
        df.to_pickle(dataframe_path)

    
