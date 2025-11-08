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
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
from utils import *
from bertopic import BERTopic
# from jinja2 import Environment, FileSystemLoader
import seaborn as sns
import pandas as pd
from html import escape
from wordcloud import WordCloud

from dash import Dash, html, dcc, callback, Output, Input, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

granularity = "sentences"
coll_name = "nomic_ai_nomic_embed_text_v2_moe__en"

data = getDocs(coll_name, granularity, include=["documents", "metadatas"])
docs = data["documents"]
metadata = data["metadatas"]
assert docs is not None
assert metadata is not None
df1 = pd.DataFrame(metadata)
X = pd.DataFrame(np.load(f"./reduced_embds/{granularity}_{coll_name}_2d.npy"), columns=["X", "Y"])
df = pd.concat([df1, X], axis=1)
# df["book"] = df["book"].astype(int, copy=False)
df["document"] = docs

app = Dash()
# fig = go.Figure(
#         go.Scattergl(x=df["X"], y=df["Y"], showlegend=True, mode="markers",
#             marker={"size":12, "color": df["book"]}, 
#             hovertemplate=None, hoverinfo="none"
#         ),
#         go.Layout(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             height=800, width=1500
#         ),
# )
# print(df["book"])
fig = px.scatter(df, x="X", y="Y",
                 height=800, color="book")
fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

fig.update_traces(
    marker={"size":12},
    hovertemplate=None,
    hoverinfo="none"
)

# Requires Dash 2.17.0 or later
app.layout = [
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Dropdown(['book', 'chapter', 'edition'], 'book', id='dropdown'),
    dcc.Graph(id='graph-content', figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id='tooltip')
]

@callback(
    Output("tooltip", "show"),
    Output("tooltip", "bbox"),
    Output("tooltip", "children"),
    Input("graph-content", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    # curve_index = int(pt["curveNumber"])
    # book = int(app.layout[1].figure['data'][curve_index]['name'])
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    # df_row = df.iloc[num]
    # book = df_row['book'] 
    # chapter = df_row['chapter']
    # doc = df_row['document']
    # if len(doc) > 300:
        # doc = doc[:100] + '...'

    children = [
        html.Div([
            # html.H2(f"Book: {book}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
            # html.H3(f"Chapter: {chapter}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
            html.P(f"{pt}"),
            # html.P(f"{df_row}")
            # html.P(f"{desc}"),
        ], style={'width': '200px', 'whiteSpace': 'normal'})
    ]
    return True, bbox, children

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown', 'value')
)
def update_graph(value):
    fig = px.scatter(df, x="X", y="Y",
                 height=800, color=value)
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    fig.update_traces(
        marker={"size":12},
        hovertemplate=None,
        hoverinfo="none"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
