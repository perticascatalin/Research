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