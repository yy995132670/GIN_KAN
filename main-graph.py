import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_graphs, optimizer, epoch):
    # 设置模型为训练模式
    model.train()

    # 获取每个 epoch 中的迭代次数
    total_iters = args.iters_per_epoch
    # 使用 tqdm 显示进度条
    pbar = tqdm(range(total_iters), unit='batch')

    # 初始化损失累计值
    loss_accum = 0
    for pos in pbar:
        # 随机选择一个批次的训练图
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        # 获取该批次的图
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # 将批次图输入模型，获取输出
        output = model(batch_graph)

        # 获取该批次的标签
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # 计算损失
        loss = criterion(output, labels)

        # 反向传播和优化
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 将损失从计算图中分离出来，转移到 CPU
        loss = loss.detach().cpu().numpy()
        # 累计损失
        loss_accum += loss

        # 更新进度条显示的描述
        pbar.set_description('epoch: %d' % (epoch))

    # 计算平均损失
    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


# 在测试时以小批次的方式传递数据以避免内存溢出（不进行反向传播）
def pass_data_iteratively(model, graphs, minibatch_size=64):
    # 设置模型为评估模式
    model.eval()
    # 初始化输出列表
    output = []
    # 获取所有图的索引
    idx = np.arange(len(graphs))
    # 以小批次的方式传递数据
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        # 获取该批次的图并传递给模型，获取输出
        output.append(model([graphs[j] for j in sampled_idx]))
    # 返回所有输出的拼接结果
    return torch.cat(output, 0)


def test(args, model, device, train_graphs, test_graphs, epoch):
    # 在训练集上进行测试
    acc_train = eval(args, model, device, train_graphs)
    # 在测试集上进行测试
    acc_test = eval(args, model, device, test_graphs)

    # 打印训练和测试的准确率
    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test


def eval(args, model, device, graphs):
    # 以小批次方式传递数据，获取输出
    output = pass_data_iteratively(model, graphs)
    # 获取所有图的标签
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    # 计算预测结果的最大值索引
    pred = output.max(1, keepdim=True)[1]
    # 计算准确率
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    return correct / len(graphs)


def run_experiment(args, device):
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(
        num_layers=args.num_layers,
        num_mlp_layers=args.num_mlp_layers,
        input_dim=train_graphs[0].node_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        final_dropout=args.final_dropout,
        learn_eps=args.learn_eps,
        graph_pooling_type=args.graph_pooling_type,
        neighbor_pooling_type=args.neighbor_pooling_type,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
        if acc_test > best_test_acc:
            best_test_acc = acc_test

    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description='GIN Variants with KAN')
    parser.add_argument('--dataset', type=str, default="IMDBBINARY", help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of MLP layers (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units (default: 64)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation. Should be less than 10. (default: 0)')
    parser.add_argument('--filename', type=str, default="results.txt", help='output file')
    parser.add_argument('--learn_eps', action="store_true", help='Whether to learn the epsilon weighting')
    parser.add_argument('--degree_as_tag', action="store_true", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", help='type of graph pooling: sum, average or max')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", help='type of neighboring pooling: sum, average or max')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs to perform')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    all_test_accs = []
    for run in range(args.num_runs):
        print(f"Run {run + 1}/{args.num_runs}")
        test_acc = run_experiment(args, device)
        all_test_accs.append(test_acc)

    avg_test_acc = np.mean(all_test_accs)
    std_test_acc = np.std(all_test_accs)

    with open(args.filename, 'a') as f:
        f.write(f"Test Accuracies: {all_test_accs}\n")
        f.write(f"Average Test Accuracy: {avg_test_acc}\n")
        f.write(f"Standard Deviation of Test Accuracy: {std_test_acc}\n\n")

    print(f"Average Test Accuracy: {avg_test_acc}")
    print(f"Standard Deviation of Test Accuracy: {std_test_acc}")

if __name__ == '__main__':
    main()
