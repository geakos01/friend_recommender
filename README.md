# 🐸

# The Team
- Gergely Ákos - EAWCFT
- Kis Milán - API8EM
- Szmida Patrik - G0HPLP

# Our chosen project: **Friend recommendation with graph neural networks**

The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs).

We aim to provide a friend recommendation service based on deep learning, more specifically: using Graph Neural Networks.
We will use the [Facebook Social Circles](https://snap.stanford.edu/data/ego-Facebook.html) dataset, which contains ego networks of Facebook users. The dataset consists of 10 different ego networks, each of them containing the friends of a single user.
We will implement the node2vec paper(https://arxiv.org/abs/1607.00653), which is a graph embedding algorithm.


# Structure

In this section we'll provide a short explanation on the files found in this repository.

#### 01_EDA.ipynb:

This notebook contains the exploratory data analysis of the dataset. We'll take a look at the dataset, and try to understand the structure of the data.

#### 02_ModelDev.ipynb:

This notebook contains the implementation of the node2vec algorithm. It is used to fine-tune the hyperparameters of the model and random-walk.

#### StreamlitFunctionDev.ipynb:

This notebook is used for inference. It has the same functions as the streamlit app, but it is easier to debug.

#### streamlit_demo.py:

It is the streamlit app. It is used for inference. It takes a graph as input and returns friend recommendations for the selected user.

#### common_functions.py:

This file contains the functions that are used in the notebooks and the streamlit app.

# Modeling Steps

1. Create random walks from the graph.
2. Create an embedding matrix from the random walks with Word2Vec.
3. Make training and test data from the embedding matrix using EdgeSplitter.
4. Process edges using some operator, that takes the embeddings of the two nodes.
5. Train a Logistic Regression model on the processed edges.
6. Evaluate the model on the test data.

# Running the pipeline

There are two main ways to run the developed model.

1. Automatically via the streamlit user interface.
  - This way you receive all important metrics like accuracy and precision score.
2. Manually via the ModelDev.ipynb notebook.
  - Here each step is divided into its own cell.
  - You may create the dataset from the facebook dataset provided.
  - The next cells deal with splitting the dataset into evaluation, test and train.
  - After that the following steps occur: Embedding creation from the train graph and distance operator creation.
  - The next step deals with evaluating which operator yielded the best results with the same training set.
  - Then the best operator is tested using the specifically designated test graph.
  - The model's AUC score is returned as the result of the test function.
  - The last cell is meant to visualize the effectiveness of the embedding by reducing the dimensionality of the
feature vectors via PCA.
  - The 2 dimensional vectors are displayed on a scatterplot.

# Related Works

- https://arxiv.org/abs/1607.00653
- https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/node2vec-link-prediction.html?fbclid=IwAR03VK37YJmiShqtjREBPMM0BAxbkjgRuOGZde8Rou12yVM_wJxQWKl3OLk

# How to Run?

You can build the application yourself and run it using a docker container by navigating to the source directory and running: `docker build -t <image_name>:<tag> .`. This will automatically build the image for you. After successful image creation, you may create
a container: `docker run -p <port_to_forward_to>:7860 <image_name>:<tag>`.

All done! You may access the streamlit application by navigating to `http://localhost:<port_to_forward_to>` in your browser of choice.
