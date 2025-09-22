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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import *


if __name__ == '__main__':
    reduction, embedders, granularities = getArg(["reduction", "granularity", "embedder"])
    r1_args: dict = reduction[0]
    r2_args = {**r1_args, "n_components":2}

    p1 = UMAP(**r1_args)
    p2 = UMAP(**r2_args)

    for emb_name in tqdm(embedders):
        coll_name = getCollName(emb_name)
        for gran in granularities:
            # embedder = SentenceTransformer(emb_name, trust_remote_code=True)
            all_data = getDocs(coll_name, gran, ["embeddings"])["embeddings"]
            data = []
            data.append(all_data)
            X = np.concat(data)
            X_5d = p1.fit_transform(X)
            X_2d = p2.fit_transform(X)
            np.save(f"./reduced_embds/{gran}_{coll_name}.npy", X_5d)
            np.save(f"./reduced_embds/{gran}_{coll_name}_2d.npy", X_2d)
