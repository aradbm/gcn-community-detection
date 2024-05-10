'''
GCN model for community detection using the hungarian algorithm for accuracy calculation.
'''
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from gcn_net import GCNNet
from communities_dataset import create_dataset, load_dataset, evaluate_spectral_clustering
from torch_geometric.data import Batch


def collate_fn(batch):
    return Batch.from_data_list(batch)


class GCNCommunityDetection:
    def __init__(self, num_nodes, num_classes, q, p, num_graphs, learning_rate=0.001,
                 epochs=100, add_permutations=False, create_new_data=True, dropout=0.5):
        # dataset parameters
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.q = q
        self.p = p
        self.num_graphs = num_graphs
        self.add_permutations = add_permutations

        # training parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.create_new_data = create_new_data

        # model, optimizer and loss function
        self.dropout = dropout
        self.model = None
        self.optimizer = None
        self.loss_func = torch.nn.CrossEntropyLoss()

        # results lists
        self.loss_list = []
        self.train_accuracy_list = []
        self.test_accuracy_list = []

    def prepare_data(self):
        num_train_graphs = int(0.8 * self.num_graphs)
        num_test_graphs = self.num_graphs - num_train_graphs
        if self.create_new_data:
            create_dataset(
                self.num_nodes, self.num_classes, self.q, self.p, num_train_graphs, "pt_train.pt", self.add_permutations)
            # create_dataset(
            #     self.num_nodes, self.num_classes, self.q, self.p, num_val_graphs, "pt_val.pt")
            create_dataset(
                self.num_nodes, self.num_classes, self.q, self.p, num_test_graphs, "pt_test.pt")
        train_dataset = load_dataset("pt_train.pt")
        test_dataset = load_dataset("pt_test.pt")
        print(f"Number of train graphs: {len(train_dataset)}")
        print(f"Number of test graphs: {len(test_dataset)}")
        # num_val_graphs = int(0.1 * self.num_graphs)
        # val_dataset = load_dataset("pt_val.pt")
        # print(f"Number of val graphs: {len(val_dataset)}")
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(
            test_dataset, batch_size=32, collate_fn=collate_fn)

        return train_loader, test_loader

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for batch in train_loader:
            batch = batch.to(device)
            # Forward pass
            pred = self.model(batch)
            # Get the predicted labels and true labels
            pred_labels = pred.max(dim=1)[1]
            true_labels = batch.y
            # Compute the confusion matrix
            n_classes = pred.size(1)
            confusion_matrix = torch.zeros(
                n_classes, n_classes, dtype=torch.int64, device=device)
            for t, p in zip(true_labels, pred_labels):
                confusion_matrix[t, p] += 1
            # Apply the Hungarian algorithm to find the best permutation
            row_ind, col_ind = linear_sum_assignment(
                -confusion_matrix.cpu().numpy())
            # Permute the predictions according to the Hungarian algorithm
            permuted_pred = pred[:, col_ind]
            # Compute the loss with permuted predictions
            loss = self.loss_func(permuted_pred, true_labels)
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def test(self, test_loader):
        self.model.eval()
        total_nodes = 0
        matched_predictions = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # Forward pass
                pred = self.model(batch)
                pred_labels = pred.max(dim=1)[1]
                n_classes = pred.size(1)
                # Compute confusion matrix
                confusion_matrix = torch.zeros(
                    n_classes, n_classes, dtype=torch.int64, device=device)
                for t, p in zip(batch.y, pred_labels):
                    confusion_matrix[t, p] += 1
                # Compute accuracy using the Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(
                    confusion_matrix.cpu().numpy(), maximize=True)
                # Compute accuracy
                matched_predictions += confusion_matrix[row_ind, col_ind].sum()
                # Update total nodes
                total_nodes += batch.y.size(0)
        accuracy = matched_predictions / total_nodes
        return accuracy.item()

    def run(self):
        train_dataset, test_dataset = self.prepare_data()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCNNet(self.num_nodes, self.num_classes,
                            self.dropout).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            loss = self.train(train_dataset)
            if epoch % 10 == 0:
                train_accuracy = self.test(train_dataset)
                test_accuracy = self.test(test_dataset)
                self.loss_list.append(loss)
                self.train_accuracy_list.append(train_accuracy)
                self.test_accuracy_list.append(test_accuracy)
                print(
                    f'Epoch: {epoch}, Train Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        test_accuracy = self.test_accuracy_list[-1]
        train_accuracy = self.train_accuracy_list[-1]
        spectral_accuracy = evaluate_spectral_clustering(
            test_dataset, self.num_nodes, self.num_classes)
        # here we print a summary of the model
        print(
            f"\n################################# Summary #################################")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of classes: {self.num_classes}")
        print(f"q: {self.q}")
        print(f"p: {self.p}")
        print(f"Number of graphs: {self.num_graphs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Dropout: {self.dropout}")
        print(f"Add permutations: {self.add_permutations}")
        print(f"Epochs: {self.epochs}")
        print(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        print(f"Final train accuracy: {train_accuracy:.4f}")
        print(
            f"Spectral clustering accuracy on test set: {spectral_accuracy:.4f}")
        print(
            f"###########################################################################\n")

        return test_accuracy, spectral_accuracy

    def plot_results(self):
        plt.plot(self.loss_list, label="Loss")
        plt.plot(self.train_accuracy_list, label="Train Accuracy")
        plt.plot(self.test_accuracy_list, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        plt.title("Loss, Train Accuracy and Test Accuracy")
        #  the epochs are jumps of 10
        plt.xticks(range(0, self.epochs, 10))

        plt.legend()
        plt.show()

    def save_model(self, path="pt_gcn_model.pt"):
        torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    # ---------- Parameters ----------
    num_nodes = 100
    num_classes = 3
    num_graphs = 800
    p = 0.9
    q = 0.1
    dropout = 0.1
    epochs = 200
    learning_rate = 0.0001
    add_permutations = True
    create_new_data = True
    # ------------------------------
    gcn_cd = GCNCommunityDetection(
        num_nodes=num_nodes, num_classes=num_classes, q=q, p=p, num_graphs=num_graphs,
        learning_rate=learning_rate, epochs=epochs, add_permutations=add_permutations,
        create_new_data=create_new_data, dropout=dropout)
    test_accuracy, spectral_accuracy = gcn_cd.run()
    gcn_cd.plot_results()
    gcn_cd.save_model()
    with open("results.csv", "a") as f:
        f.write(f"{num_nodes},{num_classes},{q},{p},{num_graphs},{learning_rate},{epochs}," +
                f"{add_permutations},{dropout},{test_accuracy:.8f},{spectral_accuracy:.8f}\n")
