'''
CustomDataset of type torch.utils.data.Dataset. It is used to create a dataset of graphs.
Contains a list of graphs. Each graph is a torch_geometric.data.Data object.

num_nodes: number of nodes in each graph
num_classes: number of classes in each graph
q: probability of connecting two nodes from different classes
p: probability of connecting two nodes from the same class
num_graphs: number of graphs in the dataset
Example of usage:
dataset = CustomDataset(num_nodes, num_classes, q, p, num_graphs)
dataset[0] # access the first graph in the dataset

each graph is a Data object, which contains:
edge_index: edge index tensor of shape [2, num_edges]
x: torch.eye of shape [num_nodes, num_nodes] (one-hot encoding of the nodes)
y: node class labels of shape [num_nodes], each node has a class label

create_dataset(num_nodes, num_classes, q, p, num_graphs, file_name) - creates a dataset of graphs
load_dataset(file_name) - loads the dataset
plot_graph(graph_data) - plots a graph from the dataset

TODO: If a graph true label is [0,1,1] for example and we predict [1,0,0], than the model successfully predicted 
        the community of the graph, but the accuracy is 0 - need to fix.
        Solution:   When running the dataset in the training, for each graph we want all permutations of the true labels.
        For example, if the true labels are [0,1,2,2], than we want to the following permutations:
        [1,2,0,0], [2,1,0,0], [0,2,1,1], [2,0,1,1], [1,0,2,2], [0,1,2,2] - instead of just [0,1,2,2].
        So the model will learn that even if the labels are permuted, the graph is the same.
'''
from itertools import permutations
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data


class CommunitiesDataset(torch.utils.data.Dataset):
    def __init__(self, num_nodes, num_classes, q, p, base_graphs, include_permutations):
        self.include_permutations = include_permutations
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.base_graphs = base_graphs
        self.q = q
        self.p = p
        self.data_list = []
        self.generate_data()

    def generate_graph(self):
        # Step 1: give the nodes a class label, pick a random number between 0 and num_classes
        y = torch.randint(self.num_classes,
                          (self.num_nodes, ), dtype=torch.long)

        # Step 2: Create edges between the nodes according to the rules, node can't be connected to itself
        edge_index = []

        # For each node, connect it to other nodes using probability p if they are in the same class,
        # and probability q if they are in different classes.
        # We create undirected graph, so we add the edge (i,j) and (j,i). No self loops.
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if y[i] == y[j] and np.random.rand() < self.p:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                elif y[i] != y[j] and np.random.rand() < self.q:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(
            edge_index, dtype=torch.long).t().contiguous()

        # Step 3: Create the data object
        data = Data(edge_index=edge_index, num_nodes=self.num_nodes, y=y)

        # Step 4.2: Cluster the graph using the classes we assigned to the nodes
        data.y = y
        data.x = torch.eye(self.num_nodes)

        return data

    def generate_data(self):
        for _ in range(self.base_graphs):
            graph = self.generate_graph()
            if self.include_permutations:
                unique_labels = graph.y.unique().tolist()
                for perm in set(permutations(unique_labels)):
                    permuted_labels = graph.y.clone()
                    for idx, label in enumerate(unique_labels):
                        permuted_labels[graph.y == label] = perm[idx]
                    self.data_list.append(
                        Data(x=graph.x, edge_index=graph.edge_index, y=permuted_labels))
            else:
                self.data_list.append(graph)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def to(self, device):
        for i in range(len(self.data_list)):
            self.data_list[i] = self.data_list[i].to(device)
        return self


def create_dataset(num_nodes, num_classes, q, p, num_graphs, file_name, include_permutations=False):
    dataset = CommunitiesDataset(num_nodes, num_classes, q,
                                 p, num_graphs, include_permutations)
    torch.save(dataset, file_name)


def load_dataset(file_name):
    # Load the dataset
    dataset = torch.load(file_name)
    return dataset


def plot_graph(graph_data):
    # Ensure edge_index is on CPU before converting to numpy
    edges_raw = graph_data.edge_index.cpu().numpy()
    # list of edges in the graph (x,y) = x is connected to y
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    # Ensure labels are on CPU before converting to numpy
    labels = graph_data.y.cpu().numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw) + 1)))
    G.add_edges_from(edges)
    plt.subplot(111)
    nx.draw(G, with_labels=True, font_weight='bold', node_color=labels)
    plt.show()


# if __name__ == '__main__':
#     num_nodes = 20
#     num_classes = 3
#     q = 0.2
#     p = 0.8
#     base_graphs = 1000

#     create_dataset(num_nodes, num_classes, q, p, base_graphs,
#                    'pt_dataset.pt', include_permutations=True)
#     dataset = load_dataset('pt_dataset.pt')
#     # graph = dataset[0]
#     # plot_graph(graph)
#     # graph = dataset[1]
#     # plot_graph(graph)
#     print(len(dataset))
