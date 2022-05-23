# Link: https://keras.io/examples/graph/gnn_citations/
# python3

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import utils
import models

# Download and read the Cora dataset
zip_file = keras.utils.get_file(
    fname   = "cora.tgz",
    origin  = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract = True)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")
citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep    = "\t",
    header = None,
    names  = ["target", "source"])
print("Citations shape:", citations.shape)
print(citations.sample(frac = 1).head())

# Process and visualize the dataset
column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep    = "\t",
    header = None,
    names  = column_names)
print("Papers shape:", papers.shape)
print(papers.sample(5).T)
print(papers.subject.value_counts())

class_values = sorted(papers["subject"].unique())
class_idx    = {name: id for id, name in enumerate(class_values)}
paper_idx    = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"]  = papers["paper_id"].apply(lambda name: paper_idx[name])
papers["subject"]   = papers["subject"].apply(lambda value: class_idx[value])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])

# utils.plot_cora_graph(papers, citations)

# Prepare training and testing data
train_data, test_data = [], []
for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac = 1)
test_data  = pd.concat(test_data).sample(frac = 1)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Set baseline model parameters
hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

feature_names = set(papers.columns) - {"paper_id", "subject"}
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features and targets as numpy arrays
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()

y_train = train_data["subject"]
y_test = test_data["subject"]

# Create the model and run the experiment
baseline_model = models.create_baseline_model(hidden_units, num_features, num_classes, dropout_rate)
baseline_model.summary()

history = models.run_experiment(baseline_model, x_train, y_train, num_epochs, batch_size, learning_rate)
# utils.display_learning_curves(history)
_, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

new_instances = utils.generate_random_instances(x_train, num_classes)
logits = baseline_model.predict(new_instances)
probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
utils.display_class_probabilities(probabilities, class_values)

# Create an edges array (sparse adjacency matrix) of shape [2, num_edges]
edges = citations[["source", "target"]].to_numpy().T
# Create an edge weights array of ones
edge_weights = tf.ones(shape=edges.shape[1])
# Create a node features array of shape [num_nodes, num_features]
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32)
# Create graph info tuple with node_features, edges, and edge_weights
graph_info = (node_features, edges, edge_weights)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)

# Need to split functionality into several files before proceeding:
# from filename import model (class)
