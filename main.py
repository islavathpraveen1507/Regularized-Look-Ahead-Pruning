import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch

from activation import get_activation
from dataset import get_dataset
from hyperparameter import get_hyperparameters
from method import get_method
from train import test, train
from utils import get_sparsity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=718, help='random seed')
    parser.add_argument('--run_name', type=str, default=None, help='custom run folder name')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--network', type=str, default='mlp', help='network name')
    parser.add_argument('--method', type=str, default='mp', help='pruning method')
    parser.add_argument('--pruning_type', type=str, default='oneshot', help='oneshot, iterative, or global')
    parser.add_argument('--pruning_iteration_start', type=int, default=1)
    parser.add_argument('--pruning_iteration_end', type=int, default=30)

    # CLI flags eliminating manual code editing
    parser.add_argument('--use_l1', action='store_true', default=False, help='enable L1 regularization')
    parser.add_argument('--lambda_l1', type=float, default=1e-4, help='L1 regularizer weight')
    parser.add_argument('--retrain', action='store_true', default=False, help='enable post-pruning retraining')
    parser.add_argument('--retrain_l1', action='store_true', default=False, help='use L1 during retraining')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, test_dataset = get_dataset(args.dataset)

    network, prune_ratios, optimizer, pretrain_iteration, \
        finetune_iteration, batch_size = get_hyperparameters(args.network, dataset=args.dataset)

    # Collision-proof folder tags
    reg_tag = f"l1_{args.lambda_l1}" if args.use_l1 else "none"
    retrain_tag = "with_retrain" if args.retrain else "no_retrain"

    default_run_name = f"{args.dataset}_{args.network}_{reg_tag}_{retrain_tag}_{args.pruning_type}_{args.seed}"
    base_folder_name = args.run_name or default_run_name
    base_path = os.path.join('./checkpoint', base_folder_name)
    os.makedirs(base_path, exist_ok=True)

    base_model_path = os.path.join(base_path, f'base_model_{reg_tag}.pth')

    if not os.path.exists(base_model_path):
        print(f"Pre-training base network: {args.network} on {args.dataset} (Reg: {reg_tag})", flush=True)
        train_acc, train_top5, train_loss, test_acc, test_top5, test_loss = train(
            train_dataset,
            test_dataset,
            network,
            optimizer,
            pretrain_iteration,
            batch_size,
            use_l1=args.use_l1,
            lambda_l1=args.lambda_l1
        )

        torch.save(network.state_dict(), base_model_path)
        with open(os.path.join(base_path, f'base_logs_{reg_tag}.txt'), 'w') as f:
            f.write(
                f'Train Loss: {train_loss:.3f}\tTest Loss: {test_loss:.3f}\t'
                f'Train Top-1: {train_acc:.2f}\tTrain Top-5: {train_top5:.2f}\t'
                f'Test Top-1: {test_acc:.2f}\tTest Top-5: {test_top5:.2f}\n'
            )
    else:
        print(f"Loading pre-trained base network from: {base_model_path}", flush=True)
        state_dict = torch.load(base_model_path, map_location='cpu')
        network.load_state_dict(state_dict)

    exp_path = os.path.join(base_path, args.method)
    os.makedirs(exp_path, exist_ok=True)

    original_network = network
    original_prune_ratio = prune_ratios
    pruning_method = get_method(args.method)

    for it in range(args.pruning_iteration_start, args.pruning_iteration_end + 1):
        print(f"\n[{args.method.upper()}] Pruning Iteration {it}/{args.pruning_iteration_end}", flush=True)

        if args.pruning_type == 'oneshot':
            network = deepcopy(original_network).cuda()
            prune_ratios = [1.0 - ((1.0 - ratio) ** it) for ratio in original_prune_ratio]
        elif args.pruning_type == 'iterative':
            prune_ratios = list(original_prune_ratio)
        else:
            raise ValueError('Unsupported pruning type')

        weights = network.get_weights()
        masks = network.get_masks()

        if 'lap_act' in args.method:
            act_rate = get_activation(network, train_dataset)
            assert len(act_rate) == len(weights) - 1
            for i in range(len(weights) - 1):
                if len(act_rate[i].shape) == 1:
                    act = act_rate[i].sqrt().view(act_rate[i].shape[0], 1).repeat([1, weights[i].shape[1]])
                    weights[i] *= act
                elif len(act_rate[i].shape) == 3:
                    act = act_rate[i].sqrt().sum(dim=1).sum(dim=1).view(act_rate[i].shape[0], 1, 1, 1).repeat(
                        [1, weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]]
                    )
                    weights[i] *= act

        if 'obd' in args.method:
            masks = pruning_method(deepcopy(network), train_dataset, prune_ratios, args.network, args.dataset)
        elif 'bn' in args.method:
            masks = pruning_method(weights, masks, prune_ratios, network.get_bn_weights())
        else:
            masks = pruning_method(weights, masks, prune_ratios)

        network.set_masks(masks)

        if args.retrain:
            print(f"Retraining pruned model for {finetune_iteration} iterations...", flush=True)
            train(
                train_dataset,
                test_dataset,
                network,
                optimizer,
                finetune_iteration,
                batch_size,
                use_l1=args.retrain_l1,
                lambda_l1=args.lambda_l1
            )

        total_surv_ratio = get_sparsity(network)
        test_top1, test_top5, test_loss = test(network, test_dataset, batch_size=batch_size)
        train_top1, train_top5, train_loss = test(network, train_dataset, batch_size=batch_size)

        print(
            f"Iter {it:02d} | Surv Ratio: {total_surv_ratio:.6f} | "
            f"Test Top-1: {test_top1:.2f}% | Test Top-5: {test_top5:.2f}% | Loss: {test_loss:.3f}",
            flush=True
        )

        with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
            f.write(
                f'{it}\t{total_surv_ratio:.6f}\t{train_loss:.3f}\t{test_loss:.3f}\t'
                f'{train_top1:.2f}\t{test_top1:.2f}\t{test_top5:.2f}\n'
            )

        checkpoint_path = os.path.join(exp_path, f'checkpoint_iter_{it}.pth')
        torch.save(network.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()