#!/bin/bash
# ==============================================================================
# Universal Benchmark Automation Suite (ImageNet-100, CIFAR-10, MNIST)
# Supports: 5 - Trail Full Matrix Batch in "One Go" OR Targeted Single Runs
# ==============================================================================

DATASET="imagenet100"
NETWORK="resnet50"
METHOD="lap"
SEED=(718 719 720 721 722)
RUN_ALL=false
USE_L1=false
RETRAIN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --network) NETWORK="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --use_l1) USE_L1=true ;;
        --retrain) RETRAIN=true ;;
        --all) RUN_ALL=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p ./console_logs

run_experiment() {
    local dset=$1
    local net=$2
    local mthd=$3
    local reg_flag=$4
    local retrain_flag=$5

    local cmd="python3 main.py --dataset $dset --network $net --method $mthd --seed $SEED"
    local log_tag="${dset}_${net}_${mthd}"

    if [ "$reg_flag" = true ]; then
        cmd="$cmd --use_l1"
        log_tag="${log_tag}_L1"
    else
        log_tag="${log_tag}_NoReg"
    fi

    if [ "$retrain_flag" = true ]; then
        cmd="$cmd --retrain"
        log_tag="${log_tag}_WithRetrain"
    else
        log_tag="${log_tag}_NoRetrain"
    fi

    echo "=========================================================="
    echo "Starting: $log_tag"
    echo "Command:  $cmd"
    echo "=========================================================="

    $cmd >> "./console_logs/${log_tag}.log" 2>&1
}

if [ "$RUN_ALL" = true ]; then
    echo "Executing Full Automated Benchmark Suite for Dataset: $DATASET"

    if [ "$DATASET" == "mnist" ]; then
        NETWORKS=("mlp")
    elif [ "$DATASET" == "cifar10" ]; then
        NETWORKS=("conv6" "resnet18" "vgg19")
    elif [ "$DATASET" == "imagenet100" ] || [ "$DATASET" == "imagenet100_dryrun" ]; then
        NETWORKS=("mlp" "conv6" "resnet50" "resnet18" "vgg19")
    else
        NETWORKS=("$NETWORK")
    fi

    METHODS=("mp" "lap" "lap_forward")
    REG_SETTINGS=(false true)
    RETRAIN_SETTINGS=(false true)

    for net in "${NETWORKS[@]}"; do
        for reg in "${REG_SETTINGS[@]}"; do
            for retrain in "${RETRAIN_SETTINGS[@]}"; do
                for mthd in "${METHODS[@]}"; do
                    run_experiment "$DATASET" "$net" "$mthd" "$reg" "$retrain"
                done
            done
        done
    done
    echo "All benchmarks completed successfully!"
else
    run_experiment "$DATASET" "$NETWORK" "$METHOD" "$USE_L1" "$RETRAIN"
fi