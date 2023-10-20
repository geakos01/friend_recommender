import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import networkx as nx
import stellargraph as sg

USED_PARAMS = {
    'p': 1.0,
    'q': 0.6,
    'dimensions': 128,
    'num_walks': 100,
    'walk_length': 30,
    'window_size': 13,
    'num_iter': 5,
    'workers': multiprocessing.cpu_count()
}


# ##################################ALGORITHM FUNCTIONS STARTS###################################

def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=USED_PARAMS['num_walks'], length=USED_PARAMS['walk_length'],
                   p=USED_PARAMS['p'], q=USED_PARAMS['q'])
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=USED_PARAMS['dimensions'],
        window=USED_PARAMS['window_size'],
        min_count=0,
        sg=1,
        workers=USED_PARAMS['workers'],
        epochs=USED_PARAMS['num_iter'],
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


# 2. training classifier
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


def link_prediction_classifier(max_iter=4000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
        clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


def run_link_prediction(binary_operator, examples_train, labels_train,
                        embedding_train, examples_model_selection, labels_model_selection):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score}


# ##################################ALGORITHM FUNCTIONS ENDS###################################


# ##################################INFERENCE FUNCTIONS STARTS###################################

# Data Loading Functions
def create_mapping_from_file(file):
    # Convert the uploaded file content to a string
    text = file.read()
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

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(scaled_connections)
    sg_graph = sg.StellarGraph.from_networkx(nx_graph)

    return sg_graph, id_mapper, reverse_id_mapper


def create_mapping_from_file_path(path):
    with open(path, 'rt') as file:
        text = file.read()

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

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(scaled_connections)
    sg_graph = sg.StellarGraph.from_networkx(nx_graph)

    return sg_graph, id_mapper, reverse_id_mapper


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

    prob_df['prob'] = model.predict_proba(processed_tmp)[:, 1]
    prob_df['pred_class'] = model.predict(processed_tmp)

    prob_df = prob_df.set_index(['first_node', 'second_node'])

    graph_edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    graph_df = pd.DataFrame(graph_edges, columns=['first_node', 'second_node'])

    graph_df = graph_df.set_index(['first_node', 'second_node'])

    graph_df['edge'] = 1

    result_df = prob_df.join(graph_df, how='left')
    result_df['edge'] = result_df['edge'].fillna(0)
    result_df['pred_edge'] = result_df['prob'].apply(lambda x: 1 if x > 0.5 else 0)

    return result_df


# Metrics Functions
def calculate_recall(df):
    return recall_score(df['edge'], df['pred_edge'])


def calculate_precision(df):
    return precision_score(df['edge'], df['pred_edge'])


def calculate_f1(df):
    return f1_score(df['edge'], df['pred_edge'])


def calculate_accuracy(df):
    return accuracy_score(df['edge'], df['pred_edge'])


def calculate_metrics(df):
    recall = calculate_recall(df)
    precision = calculate_precision(df)
    f1 = calculate_f1(df)
    accuracy = calculate_accuracy(df)

    return recall, precision, f1, accuracy


def calculate_metrics_for_threshold(df, threshold):
    df['pred_edge'] = df['prob'].apply(lambda x: 1 if x > threshold else 0)

    return calculate_metrics(df)


def find_best_threshold(df):
    thresholds = np.linspace(0.01, 1, 100)
    best_threshold = 0
    best_f1 = 0
    for threshold in thresholds:
        recall, precision, f1, accuracy = calculate_metrics_for_threshold(df, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def create_confusion_matrix(df, threshold):
    df['pred_edge'] = df['prob'].apply(lambda x: 1 if x > threshold else 0)

    return confusion_matrix(df['edge'], df['pred_edge'])


def plot_confusion_matrix(df, threshold=0.5):
    cm = create_confusion_matrix(df, threshold)

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
    predictions = predictions_df['prob'].tolist()

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


def plot_precision_recall_curve(predictions):
    precision, recall, thresholds = precision_recall_curve(predictions['edge'], predictions['prob'])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.show()


def plot_histogram(df):
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.hist(df.prob, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # Customize histogram appearance
    plt.title('Probability Distribution Histogram')  # Set the title
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines
    plt.show()


def plot_degree_distribution_log(G):
    degree_dict = dict(G.degree())
    degree = list(degree_dict.values())

    plt.figure(figsize=(8, 6))
    plt.hist(degree, bins=30, color='skyblue', edgecolor='black', alpha=0.7, log=True)
    plt.title('Degree Distribution Histogram')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# ##################################INFERENCE FUNCTIONS ENDS###################################
