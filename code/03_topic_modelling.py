#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)

import os
import numpy as np
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from bertopic.dimensionality import BaseDimensionalityReduction
from sentence_transformers import SentenceTransformer
import datamapplot
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from typing import Any, List
from utils import *

def createTopicModel(embedder: str, granularity: granType, clustering: dict[str, Any], vectorizer_model, n_topics: int,
    reduction:dict[str, Any] | None = None, topic_reduction_strategy: str | None = None, outlier_reduction_strategy: str | None = None):

    coll_name = getCollName(embedder)
    docs = getDocs(coll_name, granularity, include=["documents"])["documents"]
    assert docs is not None
    embeddings=np.load(f"./reduced_embds/{granularity}_{coll_name}.npy")
    hdbscan_model = HDBSCAN(prediction_data=True, **clustering)

    return BERTopic(
                top_n_words=10,
                calculate_probabilities=False,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                umap_model=BaseDimensionalityReduction(),
                nr_topics=n_topics
            ).fit(docs, embeddings)

if __name__ == "__main__":
    vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    for conf in tqdm(iterConfigs()):
        model_filepath = getModelFilePath(conf)
        if not os.path.exists(model_filepath):
                topic_model = createTopicModel(**{"vectorizer_model": vectorizer_model,**conf})
                topic_model.save(model_filepath)
            
        else:
            topic_model = BERTopic.load(model_filepath)