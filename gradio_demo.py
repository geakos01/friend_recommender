from stellargraph.data import EdgeSplitter
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA

import stellargraph as sg
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.metrics import confusion_matrix
import gradio as gr
import time

used_params = {
    'p': 1.0,
    'q': 0.6,
    'dimensions': 128,
    'num_walks': 100,
    'walk_length': 30,
    'window_size': 13,
    'num_iter': 5,
    'workers': multiprocessing.cpu_count()
}

def create_mapping(file_content):
    # Convert the uploaded file content to a string
    text = file_content.decode('utf-8')

    # Feldolgozom a sorokat
    connections = []
    from_list = []
    to_list = []
    for row in text.split('\n')[:-1]:
        a, b = [int(x) for x in row.split(' ')]
        connections.append((a, b))

        from_list.append(a)
        to_list.append(b)

    id_mapper = {}
    reverse_id_mapper = {}
    counter = 0
    for elem in from_list:
        if elem not in id_mapper.values():
            id_mapper[counter] = elem
            reverse_id_mapper[elem] = counter
            counter += 1

    for elem in to_list:
        if elem not in id_mapper.values():
            id_mapper[counter] = elem
            reverse_id_mapper[elem] = counter
            counter += 1

    scaled_connections = []
    for first, second in connections:
        first_scaled = reverse_id_mapper[first]
        second_scaled = reverse_id_mapper[second]

        scaled_connections.append((first_scaled, second_scaled))

    return id_mapper, reverse_id_mapper, scaled_connections

def create_graph(connections):
    graph = nx.Graph()
    graph.add_edges_from(connections)
    graph = sg.StellarGraph.from_networkx(graph)

    return graph

def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=used_params['num_walks'], length=used_params['walk_length'], p=used_params['p'], q=used_params['q'])
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=used_params['dimensions'],
        window=used_params['window_size'],
        min_count=0,
        sg=1,
        workers=used_params['workers'],
        epochs=used_params['num_iter'],
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding

def operator_l2(u, v):
    return (u - v) ** 2


def link_prediction_classifier(max_iter=4000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples]

def train_link_prediction_model(
        link_examples, link_labels, get_embedding, binary_operator
):
    # Ezt akár lehet módosítani is más algoritmussal
    clf = link_prediction_classifier()

    # Itt távolságot számol a start és end pont embeddingje között
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    # Majd arra fitteli a modelt
    clf.fit(link_features, link_labels)
    return clf

def create_result_df(graph, model, embedding, operator):
    nodes = list(graph.nodes())
    nodes.sort()

    first_node = []
    second_node = []
    for node1 in nodes:
        for node2 in nodes:
            if node1 < node2:
                first_node.append(node1)
                second_node.append(node2)


    prob_df = pd.DataFrame({'first_node': first_node, 'second_node': second_node})
    processed_tmp = link_examples_to_features(prob_df.values, embedding, operator)

    prob_df['prob'] = model.predict_proba(processed_tmp)[:,1]
    prob_df['class'] = model.predict(processed_tmp)

    prob_df = prob_df.set_index(['first_node', 'second_node'])

    graph_edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    graph_df = pd.DataFrame(graph_edges, columns=['first_node', 'second_node'])

    graph_df = graph_df.set_index(['first_node', 'second_node'])

    graph_df['edge'] = 1

    result_df = prob_df.join(graph_df, how = 'left')
    result_df['edge'] = result_df['edge'].fillna(0)
    result_df['pred_edge'] = result_df['prob'].apply(lambda x: 1 if x > 0.5 else 0)


    return result_df

def process_file(file):
    # Read the uploaded file content
    file_content = file.read()

    # Create mapping, graph, and predictions
    mapper, reverse_mapper, graph_edges = create_mapping(file_content)
    G = create_graph(graph_edges)
    edge_splitter = EdgeSplitter(G)
    splitted_graph, X, y = edge_splitter.train_test_split(p=0.2, method="global")
    embedding_all = node2vec_embedding(splitted_graph, "Graph")

    # Simulate processing by sleeping for a few seconds
    time.sleep(5)  # Adjust the duration as needed

    model = train_link_prediction_model(X, y, embedding_all, operator_l2)
    result_df = create_result_df(splitted_graph, model, embedding_all, operator_l2)

    return result_df.to_csv()

# Gradio Interface
iface = gr.Interface(
    fn=process_file,
    inputs=gr.inputs.File(label="Upload Graph File"),
    outputs=gr.outputs.File(label="Result DataFrame CSV"),
    title="Graph Processing App",
    description="Upload a graph file and get the processed result as a CSV file.",
)

iface.launch()


# mapper, reverse_mapper, graph_edges = create_mapping(file)
# G = create_graph(graph_edges)

# edge_splitter = EdgeSplitter(G)
# splitted_graph, X, y = edge_splitter.train_test_split(p=0.2, method="global")

# embedding_all = node2vec_embedding(splitted_graph, "Graph") # TODO: Megkérdezni valaki, hogy G-vel miért nem működik, nem értem

# model = train_link_prediction_model(X, y, embedding_all, operator_l2)

# predictions = create_result_df(splitted_graph, model, embedding_all, operator_l2)
#%%
