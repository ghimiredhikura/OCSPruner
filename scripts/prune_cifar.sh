run_cifar_resnet()
{
    python main_cifar.py \
    --mode $1 --model $2 --method $4 \
    --dataset $3 --batch-size 128 \
    --pruning-stability-thresh 0.999 --target-flop-rr $5 --layer-prune-limit $8 \
    --total-epochs 300 --learning-rate '(MultiStepLR, 0.1, [0.3|0.6|0.8|0.9], 0.2)' \
    --sl-start-epoch 30 --sl-end-epoch 110 \
    --reg $6 --reg-delta 1e-4 --reg-add-interval 1 \
    --prune-monitor-start-epoch 2 --output-dir $7
}

run_cifar_resnet prune "resnet32" "cifar10" "group_norm_sl_ocspruner" 0.5500 1e-4 "pruning_result_cifar" 0.75