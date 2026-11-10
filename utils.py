import torch
import torch.nn as nn
from network.masked_modules import MaskedLinear, MaskedConv2d


def is_base_module(m):
    return isinstance(m, (nn.Linear, nn.Conv2d))


def is_masked_module(m):
    return isinstance(m, (MaskedLinear, MaskedConv2d))


def is_batch_norm(m):
    return isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))


def get_sparsity(model):
    """
    Computes parameter survival ratio and sparsity strictly across masked modules
    (MaskedLinear and MaskedConv2d), matching the original log tables.
    """
    nonzero = 0
    total = 0

    for name, m in model.named_modules():
        if is_masked_module(m):
            p = m.mask
            nz_count = (p != 0).type(torch.float).sum().item()
            total_count = p.numel()
            nonzero += nz_count
            total += total_count

            print(
                f'{name:20} | nonzeros = {int(nz_count):7}/{total_count} '
                f'({100 * nz_count / total_count:6.2f}%) | '
                f'total_pruned = {int(total_count - nz_count):7} | '
                f'shape = {list(p.data.shape)}'
            )

    survival_ratio = nonzero / max(1, total)
    pruned_percent = 100 * (total - nonzero) / max(1, total)
    compression_rate = total / max(1e-8, nonzero)

    print(
        f'surv: {int(nonzero)}, pruned: {int(total - nonzero)}, '
        f'total: {int(total)}, Comp. rate: {compression_rate:10.2f}x '
        f'({pruned_percent:6.2f}% pruned)'
    )

    return survival_ratio