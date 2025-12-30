#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.disable(logging.WARN)

import os
import numpy as np
from umap import UMAP
from umap.umap_ import nearest_neighbors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import *


if __name__ == '__main__':
    # reduction, embedders, granularities = getArg(["reduction", "granularity", "embedder"])
    # reduction = {
    #     "n_neighbors": 10, "n_components":5, "min_dist":0.0,
    #     "low_memory": False, "metric":"cosine", "random_state":42
    # }
    # embedders = ["sentence-transformers/LaBSE"]
    # granularities = ["topics"]



    for conf in tqdm(iterConfigs(["granularity"])):
        # emb_name = conf["embedder"]
        gran = conf["granularity"]
        coll_name = "jinaai_jina_embeddings_v3__en_2"
        # coll_name = getCollName(emb_name)
        all_data = getDocs(coll_name, gran, ["embeddings"])["embeddings"]
        data = []
        data.append(all_data)
        X = np.concat(data)
        max_k_nn = max(getArg("reduction")["n_neighbors"])
        print(f"Pre-computing k-nearest neighbors for k_max={max_k_nn}...")
        precomputed_knn = nearest_neighbors(
            X, n_neighbors=max_k_nn, metric="cosine", metric_kwds=None, angular=True,
            low_memory=False, random_state=42
        )
        print("Done!")

        for conf2 in iterConfigs(["reduction"]):
            reduction_filepath = getReductionFilePath({**conf, **conf2, "coll_name": coll_name})
            if not os.path.exists(reduction_filepath):
                reduction = conf2["reduction"]
                p1 = UMAP(**reduction, precomputed_knn=precomputed_knn)
                dimens = reduction["n_components"]
                X_reduced = p1.fit_transform(X)
                np.save(reduction_filepath, X_reduced)
            # for emb_name in tqdm(conf2):
            #     for gran in granularities:
                    # embedder = SentenceTransformer(emb_name, trust_remote_code=True)
