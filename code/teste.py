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
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import datamapplot
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from typing import Any, List
from utils import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly import express as px
import cairosvg

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=100, scale=10)
    wc = wc.generate_from_frequencies(text)
    return wc.to_svg(embed_font=True, optimize_embedded_font=False)
    # cairosvg.svg2png(img, write_to=filename)

def create_wordcloud2(model, topic, ax):
    texts = model.get_representative_docs(topic)
    text = " ".join(texts)
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

model_filepath = "./models/topics_intfloat_multilingual_e5_large_instruct__en_10"

topic_model = BERTopic.load(model_filepath)
docs = getDocs("nomic_ai_nomic_embed_text_v2_moe__en", "topics", include=["documents", "metadatas"])["documents"]
assert docs is not None
ctfidf = ClassTfidfTransformer(True, True)
# countVectorizer = CountVectorizer(stop_words='english')
# topic_model.update_topics(docs, top_n_words=100, vectorizer_model=countVectorizer, ctfidf_model=ctfidf)
topic_info = topic_model.get_topic_info()

for topic in topic_info.index:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
                # ax1.set_title('rep')
                # ax2.set_title('docs')
                img = create_wordcloud(topic_model, topic)
                with open(f"t_{topic}.svg", 'w') as file:
                    file.write(img)
                # create_wordcloud2(topic_model, topic, ax2)
                # fig.savefig(f"testefig_t_{topic}.png", bbox_inches='tight')
                # fig.savefig(f"visualizations/images/wordclouds/{granularity}_{coll_name}_{n_topics}_t_{topic}.png")
                # plt.close()
                # plt.clf()
                # plt.cla()

    