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


def _safe_input_channels(x, network):
    """
    Safely expands 1-channel grayscale input to 3-channel RGB only if
    the network's first layer expects 3 channels.
    """
    if x.dim() == 4 and x.shape[1] == 1:
        expected_c = None
        if hasattr(network, 'conv1'):
            expected_c = getattr(network.conv1, 'in_channels', None)
        elif hasattr(network, 'features') and len(network.features) > 0:
            expected_c = getattr(network.features[0], 'in_channels', None)

        if expected_c == 3:
            x = x.repeat(1, 3, 1, 1)
    return x


def train(train_dataset, test_dataset, network, optimizer, num_iterations, batch_size,
          print_step=1000, use_l1=False, lambda_l1=1e-4):
    network.train()
    batch_sampler = BatchSampler(train_dataset, num_iterations, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    opt = optimizer(network.parameters())

    for i, (x, y) in enumerate(train_loader):
        network.train()
        x = x.cuda()
        y = y.cuda()

        x = _safe_input_channels(x, network)

        opt.zero_grad()
        out = network(x)
        loss = F.cross_entropy(out, y)

        if use_l1:
            l1_reg = sum(torch.sum(torch.abs(param)) for param in network.parameters())
            loss = loss + lambda_l1 * l1_reg

        loss.backward()
        opt.step()

        if (i + 1) % print_step == 0:
            top1_val, top5_val, val_loss = test(network, test_dataset, batch_size=batch_size)
            print(f'Steps: {i + 1}/{num_iterations}\tVal Loss: {val_loss:.3f}\tTop-1: {top1_val:.2f}%\tTop-5: {top5_val:.2f}%', flush=True)
            network.train()

    top1_train, top5_train, train_loss = test(network, train_dataset, batch_size=batch_size)
    top1_test, top5_test, test_loss = test(network, test_dataset, batch_size=batch_size)
    print(f'Final Train Loss: {train_loss:.3f} | Top-1: {top1_train:.2f}% | Top-5: {top5_train:.2f}%')
    print(f'Final Test Loss:  {test_loss:.3f} | Top-1: {top1_test:.2f}% | Top-5: {top5_test:.2f}%')

    return top1_train, top5_train, train_loss, top1_test, top5_test, test_loss


def test(network, dataset, batch_size=64):
    network.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    correct_top1 = 0
    correct_top5 = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()

            x = _safe_input_channels(x, network)

            out = network(x)
            num_classes = out.size(1)
            k = min(5, num_classes)

            _, pred_topk = out.topk(k, dim=1, largest=True, sorted=True)
            correct_top5 += pred_topk.eq(y.view(-1, 1).expand_as(pred_topk)).sum().item()

            _, pred_top1 = out.max(1)
            correct_top1 += pred_top1.eq(y).sum().item()

            total_loss += F.cross_entropy(out, y, reduction='sum').item()

    num_samples = len(dataset)
    top1_acc = correct_top1 / num_samples * 100.0
    top5_acc = correct_top5 / num_samples * 100.0
    avg_loss = total_loss / num_samples

    return top1_acc, top5_acc, avg_loss