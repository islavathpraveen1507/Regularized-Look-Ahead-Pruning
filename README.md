# Regularized-Look-Ahead-Pruning
This repository introduces $L_1$-Regularized LookAhead Pruning (L1-LAP).
<img width="1766" height="426" alt="image" src="https://github.com/user-attachments/assets/6c8ee83b-b619-4856-88d3-8f0151438379" />

# Overview
Modern deep neural networks are highly over-parameterized, making them difficult to deploy on resource-constrained edge devices. While Magnitude-Based Pruning (MBP) is a popular compression technique, it often struggles at extremely high sparsity levels. This repository introduces $L_1$-Regularized Lookahead Pruning (L1-LAP). By extending the data-free, one-shot Lookahead Pruning (LAP) algorithm—which minimizes multi-layer Frobenius distortion - we introduce an $L_1$ regularization penalty during the training phase. Based on the lemma that a pruning mask can only become optimally sparse if the underlying weight tensor is intrinsically sparse, this pre-pruning regularization forces weight sparsity. This allows the subsequent Lookahead score generation to safely disconnect highly redundant pathways without suffering catastrophic accuracy drops at extreme compression rates(>100x).

# Supported Architectures & Datasets
1. Architectures: ResNet-18, ResNet-50, VGG-19, Conv-6, MLP

2. Datasets: CIFAR-10, ImageNet-100, MNIST

# Reproducing 5-Trial Benchmarks
To replicate the 5-trial statistical benchmarks (Mean $\pm$ Std) for ResNet-18, ResNet-50, Vgg-19, Conv6, MLP on CIFAR-10, ImageNet100, MNIST without human intervention, use the provided automation suite. A quick look to run_suite.sh for clear understanding.

Execute the 5-seed benchmark on a designated GPU

# Run experiments

```
chmod +x run_suite.sh
CUDA_VISIBLE_DEVICES=1/0 

```

Run MNIST experiments (MNIST)
```
nohup ./run_suite.sh --dataset mnist --all > master_5trials.log 2>&1 &

```

Run MNIST experiments (CIFAR10)
```
nohup ./run_suite.sh --dataset cifar10 --all > master_5trials.log 2>&1 &

```
Run MNIST experiments (IMAGENET100)
```
nohup ./run_suite.sh --dataset imagenet100 --all > master_5trials.log 2>&1 &

```


The suite automatically computes results for four distinct regimes across seeds 718–722: For Networks ResNet-18, ResNet-50, Conv-6, mlp

* No Regularization, No Retraining

* No Regularization, With Retraining

* $L_1$ Regularization, No Retraining

* $L_1$ Regularization, With Retraining





