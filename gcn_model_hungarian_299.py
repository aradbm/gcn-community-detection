'''
Simple model for community detection using the hungarian algorithm for accuracy calculation.
'''
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from gcn_net import GCNNet
from communities_dataset import create_dataset, load_dataset


class GCNCommunityDetection:
    def __init__(self, num_nodes, num_classes, q, p, num_graphs, learning_rate=0.001, epochs=100, is_data_augmentation=False, create_new_data=True):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.q = q
        self.p = p
        self.num_graphs = num_graphs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_data_augmentation = is_data_augmentation
        self.create_new_data = create_new_data

        self.model = None
        self.optimizer = None
        self.loss_func = torch.nn.CrossEntropyLoss()

        self.loss_list = []
        self.accuracy_list = []

    def prepare_data(self):
        # create train_dataset, val_dataset, test_dataset, only train with augmented data
        train_dataset = []
        val_dataset = []
        test_dataset = []
        num_train_graphs = int(0.8 * self.num_graphs)
        num_val_graphs = int(0.1 * self.num_graphs)
        num_test_graphs = self.num_graphs - num_train_graphs - num_val_graphs
        if self.create_new_data:
            create_dataset(
                self.num_nodes, self.num_classes, self.q, self.p, num_train_graphs, "pt_train.pt", self.is_data_augmentation)
            create_dataset(
                self.num_nodes, self.num_classes, self.q, self.p, num_val_graphs, "pt_val.pt")
            create_dataset(
                self.num_nodes, self.num_classes, self.q, self.p, num_test_graphs, "pt_test.pt")
        train_dataset = load_dataset("pt_train.pt")
        val_dataset = load_dataset("pt_val.pt")
        test_dataset = load_dataset("pt_test.pt")
        print(f"Number of train graphs: {len(train_dataset)}")
        print(f"Number of val graphs: {len(val_dataset)}")
        print(f"Number of test graphs: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset):
        self.model.train()
        total_loss = 0
        for data in train_dataset:
            # Forward pass
            pred = self.model(data)
            loss = self.loss_func(pred, data.y)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_dataset)

    def test(self, test_dataset):
        self.model.eval()
        total_nodes = 0
        matched_predictions = 0
        for data in test_dataset:
            # Forward pass
            pred = self.model(data)
            pred_labels = pred.max(dim=1)[1]
            n_classes = pred.size(1)
            # Compute confusion matrix
            confusion_matrix = torch.zeros(
                n_classes, n_classes, dtype=torch.int64)
            for t, p in zip(data.y.cpu().numpy(), pred_labels.cpu().numpy()):
                confusion_matrix[t, p] += 1
            # Compute accuracy using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(
                confusion_matrix.cpu().numpy(), maximize=True)
            # Compute accuracy
            matched_predictions += confusion_matrix[row_ind, col_ind].sum()
            # Update total nodes
            total_nodes += data.y.size(0)
        accuracy = matched_predictions / total_nodes
        return accuracy.item()

    def run(self):
        train_dataset, val_dataset, test_dataset = self.prepare_data()

        self.model = GCNNet(self.num_nodes, self.num_classes)
        self.model = self.model.to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            loss = self.train(train_dataset)
            self.loss_list.append(loss)
            accuracy = self.test(val_dataset)
            self.accuracy_list.append(accuracy)
            print(
                f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy on val_set: {accuracy:.4f}')

        test_accuracy = self.test(test_dataset)
        # here we print a summary of the model
        print(f"\n############# Summary #############")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of classes: {self.num_classes}")
        print(f"q: {self.q}")
        print(f"p: {self.p}")
        print(f"Number of graphs: {self.num_graphs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        # print number of parameters
        print(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Accuracy on the test_set: {test_accuracy:.4f}")
        print(f"###################################\n")

    def plot_results(self):
        plt.plot(self.loss_list, label="Loss")
        plt.plot(self.accuracy_list, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        plt.title("Loss and Accuracy over Epochs")

        plt.legend()
        plt.show()

    def save_model(self, path="pt_gcn_model.pt"):
        torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    num_nodes = 100
    num_classes = 2
    gcn_cd = GCNCommunityDetection(
        num_nodes=num_nodes, num_classes=num_classes, q=0.2, p=0.8, num_graphs=1000, epochs=100,
        is_data_augmentation=False, learning_rate=0.0001, create_new_data=True)
    gcn_cd.run()
    gcn_cd.plot_results()
    gcn_cd.save_model()

    # Load the model, create 3 graphs and predict the communities
    model = GCNNet(num_nodes, num_classes)
    model.load_state_dict(torch.load("pt_gcn_model.pt"))
    model.eval()
    for _ in range(3):
        data = create_dataset(num_nodes, num_classes,
                              0.1, 0.9, 1, "pt_new_graph.pt")
        data = load_dataset("pt_new_graph.pt")
        out = model(data[0])
        pred_labels = out.max(dim=1)[1]
        print("True labels: ", data[0].y)
        print("Predicted labels: ", pred_labels)
        # Compute accuracy using the Hungarian algorithm
        n_classes = out.size(1)
        confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
        for t, p in zip(data[0].y.cpu().numpy(), pred_labels.cpu().numpy()):
            confusion_matrix[t, p] += 1
        row_ind, col_ind = linear_sum_assignment(
            confusion_matrix.cpu().numpy(), maximize=True)
        matched_predictions = confusion_matrix[row_ind, col_ind].sum()
        total_nodes = data[0].y.size(0)
        accuracy = matched_predictions / total_nodes
        print("Accuracy: ", accuracy.item())
