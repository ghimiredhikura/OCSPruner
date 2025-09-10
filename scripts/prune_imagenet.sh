
imagenet_pruning_cos()
{
    #OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 --nnodes=1 --master_port 18116 \
    python main_imagenet.py --model $1 --batch-size 256 --prune \
    --cache-dataset --method group_norm_sl_ocspruner --output-dir $3 \
    --global-pruning --print-freq 1500 --workers 8 \
    --pruning-stability-thresh 0.999 --layer-prune-limit 0.75 \
    --total-epochs 130 --learning-rate '('Cosine', 0.1)' \
    --sl-start-epoch 10 --sl-end-epoch 50 --prune-monitor-start-epoch 2 \
    --target-flops-rr $2 --reg 1e-4 --reg-delta 1e-4 --reg-add-interval 2
}

imagenet_pruning_cos resnet18 0.4500 'imagenet_pruning/resnet18_0.4500'