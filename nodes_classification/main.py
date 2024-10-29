import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GIN_KAN, GIN, GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_and_evaluate(model, features, adj, labels, idx_train, idx_val, device, model_name, patience=15):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0  # 用于记录验证集损失未改善的次数

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        train_losses.append(loss_train.item())

        # Calculate training accuracy
        _, preds = output[idx_train].max(1)
        correct = preds.eq(labels[idx_train]).sum().item()
        train_acc = correct / idx_train.size(0)
        train_accuracies.append(train_acc)

        # Calculate validation loss and accuracy
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            val_losses.append(loss_val.item())

            _, preds = output[idx_val].max(1)
            correct = preds.eq(labels[idx_val]).sum().item()
            val_acc = correct / idx_val.size(0)
            val_accuracies.append(val_acc)

        model.train()

        # 早停机制：如果验证损失没有改善，计数加1；否则，重置计数
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1} for {model_name}')
            break

        if epoch % 10 == 0:
            print(f'{model_name} Epoch: {epoch + 1:04d}',
                  f'loss_train: {loss_train.item():.4f}',
                  f'train_acc: {train_acc:.4f}',
                  f'loss_val: {loss_val.item():.4f}',
                  f'val_acc: {val_acc:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--seed', type=int, default=10, help='seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    data = Dataset(root='./dataset', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert sparse matrices to dense
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(labels).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    models = {

        'GIN_KAN': GIN_KAN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0.5,
                           with_relu=False, with_bias=False, weight_decay=5e-4, device=device),
        'GIN': GIN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0.5, with_relu=False,
                   with_bias=False, weight_decay=5e-4, device=device),
        # 'GCN': GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0.5, with_relu=False,
        #            with_bias=False, weight_decay=5e-4, device=device),
    }

    # Store loss and accuracy values for each model
    train_loss_values = {}
    val_loss_values = {}
    train_accuracy_values = {}
    val_accuracy_values = {}

    for model_name, model in models.items():
        model.attention = False  # Explicitly define the attention attribute
        train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(model, features, adj, labels, idx_train, idx_val, device, model_name)
        train_loss_values[model_name] = train_losses
        val_loss_values[model_name] = val_losses
        train_accuracy_values[model_name] = train_accuracies
        val_accuracy_values[model_name] = val_accuracies

    # Plot the loss curves
    plt.figure()
    for model_name, train_losses in train_loss_values.items():
        plt.plot(range(len(train_losses)), train_losses, label=f'{model_name} Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train Loss of Different Models')
    plt.show()

    plt.figure()
    for model_name, val_losses in val_loss_values.items():
        plt.plot(range(len(val_losses)), val_losses, label=f'{model_name} Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss of Different Models')
    plt.show()

    # Plot the accuracy curves
    plt.figure()
    for model_name, train_accuracies in train_accuracy_values.items():
        plt.plot(range(len(train_accuracies)), train_accuracies, label=f'{model_name} Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train Accuracy of Different Models')
    plt.show()

    plt.figure()
    for model_name, val_accuracies in val_accuracy_values.items():
        plt.plot(range(len(val_accuracies)), val_accuracies, label=f'{model_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy of Different Models')
    plt.show()

    # Evaluate each model
    for model_name, model in models.items():
        model.eval()
        output = model(features, adj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print(f"{model_name} Test set results: accuracy= {acc_test.item():.4f}")

if __name__ == '__main__':
    main()
