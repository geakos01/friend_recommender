import streamlit as st
import pandas as pd  # Import Pandas for data processing

# Function to process the uploaded file (you can replace this with your processing logic)
def process_edges_file(uploaded_file):
    with st.spinner("Processing..."):
        # You can add your processing code here
        st.text("Processing in progress...")
        # Simulate processing (remove this line in your actual processing code)
        import time
        time.sleep(5)
        st.success("Processing completed!")

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
st.set_option('deprecation.showPyplotGlobalUse', False)

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
    # text = file_content.decode('utf-8')
    text = file_content.read()
    text = text.decode('utf-8')

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

    return scaled_connections, id_mapper, reverse_id_mapper

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
    prob_df['pred_class'] = model.predict(processed_tmp)

    prob_df = prob_df.set_index(['first_node', 'second_node'])

    graph_edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    graph_df = pd.DataFrame(graph_edges, columns=['first_node', 'second_node'])

    graph_df = graph_df.set_index(['first_node', 'second_node'])

    graph_df['edge'] = 1

    result_df = prob_df.join(graph_df, how = 'left')
    result_df['edge'] = result_df['edge'].fillna(0)
    result_df['pred_edge'] = result_df['prob'].apply(lambda x: 1 if x > 0.5 else 0)


    return result_df

def plot_confusion_matrix(df, real, predicted):
    cm = confusion_matrix(df[real], df[predicted])

    class_labels = ['Negative Edge', 'Edge']

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(predictions_df):
    # Extract the 'Real', 'Pred', and 'Prob' columns
    ground_truth_values = predictions_df['edge'].tolist()
    predictions = predictions_df['pred_edge'].tolist()

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(ground_truth_values, predictions)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = roc_auc_score(ground_truth_values, predictions)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_histogram(df):
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.hist(df.prob, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # Customize histogram appearance
    plt.title('Probability Distribution Histogram')  # Set the title
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines
    plt.show()
@st.cache_data
def make_predictions(df, node, recommendation_number=5):
    offers = df[(df['edge'] == 0) & (df['pred_edge'] == 1)].reset_index()

    filtered_offers = offers[(offers['first_node'] == node) | (offers['second_node'] == node)]

    filtered_offers['neighbor'] = filtered_offers.apply(lambda x: x['first_node'] if x['first_node'] != node else x['second_node'], axis = 1).astype('int')

    st.write(list(filtered_offers.sort_values('prob', ascending = False)['neighbor'][:recommendation_number]))



@st.cache_data
def main_process(file):
    graph_edges, mapper, reverse_mapper = create_mapping(file)
    # G, mapper, reverse_mapper = read_graph(file)
    G = create_graph(graph_edges)

    edge_splitter = EdgeSplitter(G)
    splitted_graph, X, y = edge_splitter.train_test_split(p=0.2, method="global")

    embedding_all = node2vec_embedding(splitted_graph, "Graph") # TODO: Megkérdezni valaki, hogy G-vel miért nem működik, nem értem

    model = train_link_prediction_model(X, y, embedding_all, operator_l2)

    predictions = create_result_df(splitted_graph, model, embedding_all, operator_l2)

    st.write('Finished Preprocessing')
    return predictions.reset_index()

st.title("EDGES File Upload and Processing App")

# Create a file uploader widget for EDGES files (you can specify other file types)
uploaded_file = st.file_uploader("Upload an EDGES file", type=[".edges"])


if uploaded_file is not None:
    predictions = None
    node_id = None

    # Create button objects with unique keys
    # start_processing_button = st.button("Start Processing", key="embedding_button")


    # if start_processing_button:
    predictions = main_process(uploaded_file)

    plot_histogram(predictions)
    st.pyplot()

    plot_confusion_matrix(predictions, 'edge', 'pred_edge')
    st.pyplot()

    plot_roc_curve(predictions)
    st.pyplot()



    if predictions is not None:
        # Ask for a node_id
        node_id = st.number_input("Enter a Node ID for Recommendations", min_value=0, max_value=max([max(predictions['first_node']), max(predictions['first_node'])]))

        st.write(f'Selected Node ID {node_id}')

    get_recommendations_button = st.button("Get Recommendations", key="recommendation_button")

    if get_recommendations_button and predictions is not None:
        make_predictions(predictions, node_id)

        # Display the other plots and charts here

