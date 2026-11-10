from functools import partial
import torch.optim as optim

from network import (
    MaskedConv6,
    MaskedLeNet,
    MaskedMLP,
    MaskedResNet18,
    MaskedResNet50,
    MaskedVGG11,
    MaskedVGG16,
    MaskedVGG19,
)


def get_dataset_info(dataset):
    """
    Returns (num_classes, in_channels, spatial_dimension)
    for dynamic network parameterization.
    """
    if dataset in ['mnist', 'fmnist']:
        return 10, 1, 28
    elif dataset in ['cifar10', 'svhn']:
        return 10, 3, 32
    elif dataset == 'cifar100':
        return 100, 3, 32
    elif dataset == 'tiny-imagenet':
        return 200, 3, 64
    elif dataset in ['imagenet100', 'imagenet100_dryrun']:
        return 100, 3, 224
    else:
        return 10, 3, 32


def get_hyperparameters(network_type, dataset='cifar10'):
    num_classes, in_channels, img_dim = get_dataset_info(dataset)
    is_dryrun = 'dryrun' in dataset

    if network_type == 'mlp':
        in_features = in_channels * img_dim * img_dim
        network = MaskedMLP([in_features, 500, 500, 500, 500, num_classes]).cuda()
        prune_ratios = [.5, .5, .5, .5, .25]
        optimizer = partial(optim.Adam, lr=0.0012)
        pretrain_iteration = 100 if is_dryrun else 50000
        finetune_iteration = 50 if is_dryrun else 50000
        batch_size = 16 if img_dim >= 224 else 60

    elif network_type == 'lenet':
        network = MaskedLeNet().cuda()
        prune_ratios = [.2, .2, .3, .3, .15]
        optimizer = partial(optim.Adam, lr=0.0012)
        pretrain_iteration = 100 if is_dryrun else 50000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 60

    elif network_type == 'conv6':
        network = MaskedConv6(num_classes=num_classes, in_channels=in_channels).cuda()
        prune_ratios = [.15, .15, .15, .15, .15, .15, .20, .20, .10]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 30000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 64 if img_dim >= 224 else 60

    elif network_type == 'vgg11':
        network = MaskedVGG11(
            num_classes=num_classes, in_channels=in_channels, use_bn=True
        ).cuda()
        prune_ratios = [.15] * 8 + [.10]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 35000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 64 if img_dim >= 224 else 60

    elif network_type == 'vgg16':
        network = MaskedVGG16(
            num_classes=num_classes, in_channels=in_channels, use_bn=True
        ).cuda()
        prune_ratios = [.15] * 13 + [.10]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 50000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 64 if img_dim >= 224 else 60

    elif network_type == 'vgg19':
        network = MaskedVGG19(
            num_classes=num_classes, in_channels=in_channels, use_bn=True
        ).cuda()
        prune_ratios = [.15] * 16 + [.10]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 60000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 64 if img_dim >= 224 else 60

    elif network_type == 'resnet18':
        # Unchanged: CIFAR-style stem (imagenet_stem defaults to False)
        network = MaskedResNet18(
            num_classes=num_classes, in_channels=in_channels
        ).cuda()
        prune_ratios = [0] + [.15] * 16 + [.10]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 35000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 64 if img_dim >= 224 else 60

    elif network_type == 'resnet50':
        # For 224x224 inputs use the standard ImageNet stem (7x7 stride-2 conv + max-pool).
        # The CIFAR-style stem keeps full resolution through layer1 (~16x more activation
        # memory), which caused the CUDA OOM. Small-image datasets keep the CIFAR stem.
        network = MaskedResNet50(
            num_classes=num_classes,
            in_channels=in_channels,
            imagenet_stem=(img_dim >= 224),
        ).cuda()
        prune_ratios = [0] + [.15] * 48 + [.10]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 60000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 128 if img_dim >= 224 else 60

    elif network_type == 'mlp_global':
        in_features = in_channels * img_dim * img_dim
        network = MaskedMLP([in_features, 500, 500, 500, 500, num_classes]).cuda()
        prune_ratios = [0.9364, 0.9679, 0.9837, 0.9916, 0.9957, 0.9977, 0.9988]
        optimizer = partial(optim.Adam, lr=0.0012)
        pretrain_iteration = 100 if is_dryrun else 50000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 16 if img_dim >= 224 else 60

    elif network_type == 'conv6_global':
        network = MaskedConv6(num_classes=num_classes, in_channels=in_channels).cuda()
        prune_ratios = [0.8938, 0.9114, 0.9261, 0.9382, 0.9483, 0.9568, 0.9638]
        optimizer = partial(optim.Adam, lr=0.0003)
        pretrain_iteration = 100 if is_dryrun else 30000
        finetune_iteration = 50 if is_dryrun else 5000
        batch_size = 64 if img_dim >= 224 else 60

    else:
        raise ValueError(f'Unknown network: {network_type}')

    return (
        network,
        prune_ratios,
        optimizer,
        pretrain_iteration,
        finetune_iteration,
        batch_size,
    )