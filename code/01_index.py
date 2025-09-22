#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logging.basicConfig()

import sys
import readline
import os
import spacy
from spacy.attrs import ORTH, NORM
from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils.batch_utils import create_batches
import json
from datetime import datetime
from rich import print
import re
import gc
import torch
from sentence_transformers import SentenceTransformer
from utils import getOpt, getCollBaseName

SETTINGS = {}

SETTINGS['en'] = {'split_normal': r'^(?=[mdclxvi])m*(c[md]|d?C{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})\.',
                'split_X': r'[ABCDEFGHIKLMNOPQRSTVX]\.',
                'encoding': '1252',
                'file_name_mask': r'book_{0}_en.txt',
                'chp_id_cleanup': lambda s: s.replace('.', '').upper(),
                'num_topic_regex': r'(\d+\.)[ A-Z]',
                'topic_id_cleanup': lambda s: re.sub('[A-Z. ]', '', s),
                'senter': 'en'}

class TextLoader:
    def __init__(self, book_dir, settings):
        self.settings = settings
        self.books = {}
        self.book_dir = book_dir        

        self.nlp_en = spacy.load('en_core_web_sm', exclude=['parser'])
        self.nlp_en.enable_pipe('senter')
        self.nlp_en.tokenizer.add_special_case('lit.', [{ORTH: 'lit.', NORM: 'literally'}])

        self.senter = self.get_sentences_en


    def load_book(self, book_id):
        if book_id == '10':
            split_chp_regex = self.settings['split_X']
        else:
            split_chp_regex = self.settings['split_normal']

        book = {}
        
        book_file_name = os.path.join(self.book_dir,
            self.settings['file_name_mask'].format(book_id) if book_id != '10' else 'book_10.txt'
        )

        #empty book if not exists
        if not os.path.exists(book_file_name):
            return book

        with open(book_file_name, 'r', encoding=self.settings['encoding']) as f:
            lines = f.readlines()

        starts = []
        chp_ids = []

        #match chapter start
        for i, line in enumerate(lines):
            m = re.match(split_chp_regex, line)
            if m is not None:
                chp_ids.append(self.settings['chp_id_cleanup'](m.group()))
                starts.append(i)

        ends = [i for i in starts[1:]]
        ends.append(len(lines))

        #select sentences in chapters, join hyphenated words
        for c, s, e in zip(chp_ids, starts, ends):
            t = ''.join(lines[s:e])
            t = t.replace('\n', ' ')
            t = re.sub(r'(\w)\- ', r'\1', t)
            book[c] = {'text': t, 'topics': {}}

        #break chapters into numbered topics
        for c, t in book.items():
            text = t['text']

            s_starts = []
            s_ends = []
            topic_ids = []

            matches = re.finditer(self.settings['num_topic_regex'], text)
            for n, match in enumerate(matches, start=1):
                ss = match.start()
                se = match.end()
                id = match.group()
                id = self.settings['topic_id_cleanup'](id)

                topic_ids.append(id)
                s_starts.append(ss)

            s_ends = [i for i in s_starts[1:]]
            s_ends.append(len(text))

            book[c]['topics']['all'] = text

            for ss, se, tid in zip(s_starts, s_ends, topic_ids):
                sent = text[ss:se]
                book[c]['topics'][tid] = {}
                book[c]['topics'][tid]['all'] = sent

                sentences = self.senter(sent)

                for sid, ssent in enumerate(sentences):
                    book[c]['topics'][tid][str(sid)] = ssent

        return book


    def get_book_structure(self):
        book_struct = {}
        book_struct['1'] = {'title_en': 'Grammar', 'title_la': 'De grammatica'}
        book_struct['2'] = {'title_en': 'Rhetoric and dialectic', 'title_la': 'De rhetorica et dialectica'}
        book_struct['3'] = {'title_en': 'Mathematics, music, astronomy', 'titla_la': 'De mathematica'}
        book_struct['4'] = {'title_en': 'Medicine', 'title_la': 'De medicina'}
        book_struct['5'] = {'title_en': 'Laws and times', 'title_la': 'De legibus et temporibus'}
        book_struct['6'] = {'title_en': 'Books and ecclesiastical offices', 'title_la': 'De libris et officiis ecclesiasticis'}
        book_struct['7'] = {'title_en': 'God, angels, and saints', 'title_la': 'De deo, angelis et sanctis'}
        book_struct['8'] = {'title_en': 'The Church and sects', 'title_la': 'De ecclesia et sectis'}
        book_struct['9'] = {'title_en': 'Languages, nations, reigns, the military, citizens, family relationships', 'title_la': 'De linguis, gentibus, regnis, militia, civibus, affinitatibus'}
        book_struct['10'] = {'title_en': 'Vocabulary', 'title_la': 'De vocabulis'}
        book_struct['11'] = {'title_en': 'The human being and portents', 'title_la': 'De homine et portentis'}
        book_struct['12'] = {'title_en': 'Animals', 'title_la': 'De animalibus'}
        book_struct['13'] = {'title_en': 'The cosmos and its parts', 'title_la': 'De mundo et partibus'}
        book_struct['14'] = {'title_en': 'The earth and its parts', 'title_la': 'De terra et partibus'}
        book_struct['15'] = {'title_en': 'Buildings and fields', 'title_la': 'De aedificiis et agris'}
        book_struct['16'] = {'title_en': 'Stones and metals', 'title_la': 'De lapidibus et metallis'}
        book_struct['17'] = {'title_en': 'Rural matters', 'title_la': 'De rebus rusticis'}
        book_struct['18'] = {'title_en': 'War and games', 'title_la': 'De bello et ludis'}
        book_struct['19'] = {'title_en': 'Ships, buildings, and clothing', 'title_la': 'De navibus aedificiis et vestibus'}
        book_struct['20'] = {'title_en': 'Provisions and various implements', 'title_la': 'Provisions and various implements'}

        return book_struct


    def get_sentences_en(self, t):
        sents = []
        d = self.nlp_en(t)
        for sent in d.sents:
            if len(sent) > 5:
                sents.append(sent.text)

        return sents


    def load_books(self):
        st = self.get_book_structure()

        for book_id in st.keys():
            self.books[book_id] = self.load_book(book_id)


class Searcher:
    def __init__(self, embedder_name):
        self.embedder_name = embedder_name
        device = "cuda" if embedder_name not in [
            'intfloat/multilingual-e5-large-instruct',
            # 'nomic-ai/nomic-embed-text-v2-moe',
        ] else "cpu"
        self.embedder = SentenceTransformer(self.embedder_name, trust_remote_code=True, device=device)
        self.coll_base_name = getCollBaseName(self.embedder_name)
        self.client = PersistentClient('isidb')
        self.documents = []


    def init_collection(self, edition):
        is_loaded = False
        try:
            collection =  self.client.get_collection(self.coll_base_name + '__' + edition)
            is_loaded = True
        except:
            collection = self.client.create_collection(self.coll_base_name + '__' + edition)

        return is_loaded, collection


    def insert_db(self, batch_ids, batch_txt, collection, metadatas):
        embeddings = self.embedder.encode(batch_txt, normalize_embeddings=True)
        batches = create_batches(self.client, list(batch_ids), list(embeddings), list(metadatas), list(batch_txt))
        for batch in tqdm(batches, desc="Batches", leave=False):
            collection.add(*batch)


    def save_to_vectordb(self, book_text, edition, collection):
        #each book
        batch_ids = []
        batch_txt = []
        metadatas = []
        for book_id, book in book_text.items():
            #each chapter in book
            for chp_id in book.keys():
                topics = book[chp_id]['topics']
                #for each topic in chapter
                for topic_id, topic in topics.items():
                    #full chapter (aka all topics in one sentence)
                    if topic_id == 'all':
                        continue
                    else:
                        for sent_id, ssent in topic.items():
                            # full topic (aka all sentences in one)
                            # if sent_id == 'all':
                            #     continue
                            # # each sentence in topic
                            # else:
                                idx = '{0}.{1}.{2}.{3}.S{4}'.format(edition, book_id, chp_id, topic_id, sent_id)
                                batch_ids.append(idx)
                                batch_txt.append(ssent)
                                metadatas.append({
                                    "edition": edition,
                                    "book": book_id,
                                    "chapter": chp_id,
                                    "topic": topic_id,
                                    "sentence": sent_id,
                                })
        self.insert_db(batch_ids, batch_txt, collection, metadatas)


    def query(self, query, n, collection):
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        data = collection.query(query_embeddings=query_emb, n_results=n)

        data_ids = data['ids'][0]
        data_docs = data['documents'][0]

        return data_ids, data_docs


    def clear_vram(self):
        torch.cuda.empty_cache()
        gc.collect()


    def __del__(self):
        del self.embedder
        self.clear_vram()


if __name__ == '__main__':
    embedders = getOpt("embedders")

    book_editions = ['en']
    loaders = {}

    print("Loading books...")

    for ed in (pbar:= tqdm(book_editions, desc='Editions', leave=True, unit='ed')):
        pbar.set_postfix_str(ed)

        loaders[ed] = TextLoader('./books', SETTINGS[ed])
        book_file = f'books_{ed}.json'

        if os.path.exists(book_file):
            with open(book_file, 'r') as f:
                loaders[ed].books = json.load(f)
        else:
            loaders[ed].load_books()

            with open(book_file, 'w') as f:
                json.dump(loaders[ed].books, f)

    embedder_collections = {}

    print()
    print('Embedding books...')

    for emb in (pbar := tqdm(embedders, desc='Embedders', leave=True, unit='emb')):
        pbar.set_postfix_str(emb)

        s = Searcher(emb)
        collections = {}

        for ed in (pbared:= tqdm(book_editions, leave=False, desc='Editions')):
            is_loaded, coll = s.init_collection(ed)
            pbared.set_postfix_str(ed)

            if not is_loaded:
                # print('Loading {0}'.format(ed))
                s.save_to_vectordb(loaders[ed].books, ed, coll)

            collections[ed] = coll

        embedder_collections[emb] = collections

