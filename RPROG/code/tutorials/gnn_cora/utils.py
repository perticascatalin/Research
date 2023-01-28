import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()

def plot_cora_graph(papers, citations):
    plt.figure(figsize = (10, 10))
    colors = papers["subject"].tolist()
    cora_graph = nx.from_pandas_edgelist(citations.sample(n = 1500))
    subjects = list(papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"])
    nx.draw_spring(cora_graph, node_size = 15, node_color = subjects)
    plt.show()

def generate_random_instances(x_train, num_instances):
    token_probability = x_train.mean(axis = 0)
    instances = []
    for _ in range(num_instances):
        probabilities = np.random.uniform(size = len(token_probability))
        instance = (probabilities <= token_probability).astype(int)
        instances.append(instance)
    return np.array(instances)

def display_class_probabilities(probabilities, class_values):
    for instance_idx, probs in enumerate(probabilities):
        print(f"Instance {instance_idx + 1}:")
        for class_idx, prob in enumerate(probs):
            print(f"- {class_values[class_idx]}: {round(prob * 100, 2)}%")