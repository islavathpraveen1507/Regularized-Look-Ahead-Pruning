import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

class BatchSampler(Sampler):
    def __init__(self, dataset, num_iterations, batch_size):
        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)), self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations


def train(train_dataset, test_dataset, network, optimizer, num_iterations, batch_size, print_step=1000):
    network.train()
    batch_sampler = BatchSampler(train_dataset, num_iterations, batch_size)  # train by iteration, not epoch
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
    optimizer = optimizer(network.parameters())  # create optimizer (argument: function)

    for i, (x, y) in enumerate(train_loader):
        network.train()  # ensure training mode at start of iteration
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        out = network(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % print_step == 0:
            test_acc, test_loss = test(network, test_dataset)
            train_acc, train_loss = test(network, train_dataset)
            print(f'Steps: {i + 1}/{num_iterations}\tTrain loss: {train_loss:.3f}\tTrain acc: {train_acc:.2f}\tTest loss: {test_loss:.3f}\tTest acc (top-5): {test_acc:.2f}')
            network.train()  # ensure model is in train mode after testing

    train_acc, train_loss = test(network, train_dataset)
    test_acc, test_loss = test(network, test_dataset)
    print(f'Train loss: {train_loss:.3f}\tTrain acc: {train_acc:.2f}\tTest loss: {test_loss:.3f}\tTest acc (top-5): {test_acc:.2f}')

    return train_acc, train_loss, test_acc, test_loss


def test(network, dataset, batch_size=64):
    network.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct_top1 = 0  # For top-1 accuracy
    correct_top5 = 0  # For top-5 accuracy
    loss = 0

    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            out = network(x)
            # Get top-5 predictions
            _, pred_top5 = out.topk(5, dim=1)
            # Get top-1 prediction
            _, pred_top1 = out.max(1)

        # Calculate top-5 accuracy
        correct_top5 += pred_top5.eq(y.view(-1, 1)).sum().item()

        # Calculate top-1 accuracy (if needed, but currently unused)
        correct_top1 += pred_top1.eq(y).sum().item()

        # Accumulate loss
        loss += F.cross_entropy(out, y, reduction='sum').item()

    # Compute top-5 accuracy
    top5_acc = correct_top5 / len(dataset) * 100.0

    # Compute average loss
    loss = loss / len(dataset)

    return top5_acc, loss
