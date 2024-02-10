'''
Simple model for spectral clustering using PyTorch Geometric.

TODO: If a graph true label is [0,1,1] for example and we predict [1,0,0], 
        than the model successfully predicted the community of the graph,
        but the accuracy is 0 - need to fix. Later, gennerlize fix.
        Potential solution: when creating the dataset,for each graph we want all permutations of the true labels.
        For example, if the true labels are [0,1,1], than we want to add the following permutation [1,0,0] to the dataset
        with the same graph.

'''
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GCNConv
from communities_dataset import create_dataset, load_dataset


class Net(torch.nn.Module):
    def __init__(self, num_nodes, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_nodes, 64)
        self.conv2 = GCNConv(64, 32)  # Add an additional layer
        self.conv3 = GCNConv(32, num_classes)  # Output layer

        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return F.softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()


def train(model, loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader):
    # calculate the accuracy of the model on the test set
    model.eval()
    correct = 0
    for data in loader:
        pred = model(data)
        pred = pred.max(dim=1)[1]
        pred_result = pred.eq(data.y).sum().item()
        correct += pred_result

    # calculate the accuracy by num of successful predictions / num of nodes
    return correct / (len(loader) * len(loader.dataset[0].y))


if __name__ == '__main__':
    # Create the dataset
    num_nodes = 30
    num_classes = 2
    q = 0.1
    p = 0.9
    num_graphs = 1000
    dataset = create_dataset(num_nodes, num_classes, q,
                             p, num_graphs, "dataset.pt")
    loader = load_dataset("dataset.pt")

    # split the dataset into train, validation and test sets (80%, 10%, 10%)
    dataset_size = len(loader.data_list)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        loader.data_list, [train_size, val_size, test_size])

    # create the loaders

    model = Net(num_nodes, num_classes)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    print(device)

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001)  # Adjust learning rate
    loss_func = torch.nn.CrossEntropyLoss()

    loss_list = []
    accuracy_list = []
    # Train the model
    for epoch in range(1000):  # Adjust number of epochs
        loss = train(model, train_dataset, optimizer, loss_func)
        loss_list.append(loss)
        accuracy = test(model, val_dataset)
        accuracy_list.append(accuracy)
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    # check the accuracy of the model on the test set
    print("Accuracy on the test set: ")
    print(test(model, test_dataset))

    ############### Test the model on a new graph #################
    # check the accuracy of the model on a new graph
    print("Accuracy on a new random graph: ")
    new_graph_data = create_dataset(
        num_nodes, num_classes, q, p, 1, "new_graph.pt")
    test_loader = load_dataset("new_graph.pt")[0]
    model.eval()
    with torch.no_grad():
        pred = model(test_loader)
        pred = pred.max(dim=1)[1]
        # we want to print the graph true class and the predicted class, than calculate the accuracy
        print("True class: ")
        print(test_loader.y)
        print("Predicted class: ")
        print(pred)
        # calculate the accuracy by num of successful predictions / num of nodes
        print("Accuracy: ")
        print(pred.eq(test_loader.y).sum().item() / num_nodes)

    # plot the loss and accuracy ,
    plt.plot(loss_list, label="Loss")
    plt.plot(accuracy_list, label="Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss and Accuracy over Epochs")
    plt.legend()
    plt.show()

    # save the model
    torch.save(model.state_dict(), "gnn_model.pt")
