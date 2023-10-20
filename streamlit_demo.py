import streamlit as st
import random
from stellargraph.data import EdgeSplitter
import stellargraph as sg
import networkx as nx
import common_functions as cf
st.set_option('deprecation.showPyplotGlobalUse', False)

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


@st.cache_data
def make_predictions(df, node, mapper, recommendation_number=5):
    offers = df[(df['edge'] == 0) & (df['pred_edge'] == 1)].reset_index()

    filtered_offers = offers[(offers['first_node'] == node) | (offers['second_node'] == node)]

    filtered_offers['neighbor'] = filtered_offers.apply(
        lambda x: x['first_node'] if x['first_node'] != node else x['second_node'], axis=1).astype('int')

    text = 'Recommended Friends: '
    for elem in filtered_offers.sort_values('prob', ascending=False)['neighbor'][:recommendation_number].values:
        text += str(mapper[elem])
        text += ', '

    st.write(text[:-2])


@st.cache_data
def create_plots(df):
    cf.plot_histogram(df)
    st.pyplot()

    cf.plot_confusion_matrix(df)
    st.pyplot()

    cf.plot_roc_curve(df)
    st.pyplot()


def get_possible_nodes(options):
    sample_num = 3  # Number of elements to sample
    possible_nodes = random.sample(options, sample_num)

    text = 'Node ID examples: '
    for elem in possible_nodes:
        text += str(elem)
        text += ', '

    text = text[:-2]

    st.write(text)

@st.cache_data
def main_process(file):
    graph_edges, mapper, reverse_mapper = create_mapping(file)
    # G, mapper, reverse_mapper = read_graph(file)
    G = create_graph(graph_edges)

    edge_splitter = EdgeSplitter(G)
    splitted_graph, X, y = edge_splitter.train_test_split(p=0.2, method="global")

    embedding_all = cf.node2vec_embedding(splitted_graph, "Graph")

    model = cf.train_link_prediction_model(X, y, embedding_all, cf.operator_l2)

    predictions = cf.create_result_df(splitted_graph, model, embedding_all, cf.operator_l2)

    st.write('Finished Preprocessing')
    return predictions.reset_index(), mapper, reverse_mapper


st.title("Graph Recommender System")

# Create a file uploader widget for EDGES files (you can specify other file types)
uploaded_file = st.file_uploader("Upload a .edges file", type=[".edges"])

if uploaded_file is not None:
    predictions = None
    node_id = None

    # if start_processing_button:
    predictions, mapper, reverse_mapper = main_process(uploaded_file)

    if predictions is not None:
        max_value = max(list(mapper.values()))

        # Ask for a node_id
        node_id = st.number_input("Enter a Node ID for making recommendations:", min_value=0, max_value=max_value)

    get_possible_nodes(list(mapper.values()))

    get_recommendations_button = st.button("Get Recommendations", key="recommendation_button")

    try:
        mapped_node_id = reverse_mapper[node_id]
        if get_recommendations_button and predictions is not None:
            make_predictions(predictions, mapped_node_id, mapper)

    except KeyError:
        if node_id != 0:
            st.write("This Node ID doesn't exist")

    recall, precision, f1, accuracy = cf.calculate_metrics(predictions)

    st.write(f"Recall: {recall}")
    st.write(f"Precision: {precision}")
    st.write(f"F1: {f1}")
    st.write(f"Accuracy: {accuracy}")

    create_plots(predictions)
