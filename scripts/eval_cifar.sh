
# ResNet56-CIFAR10
python main_cifar.py --mode eval --model resnet56 --dataset "cifar10" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar10_resnet56_ocspruner_flops_47%.pth" --output-dir "eval_result_cifar"
python main_cifar.py --mode eval --model resnet56 --dataset "cifar10" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar10_resnet56_ocspruner_flops_39%.pth" --output-dir "eval_result_cifar"

# ResNet110-CIFAR10
python main_cifar.py --mode eval --model resnet110 --dataset "cifar10" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar10_resnet110_ocspruner_flops_47%.pth" --output-dir "eval_result_cifar"
python main_cifar.py --mode eval --model resnet110 --dataset "cifar10" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar10_resnet110_ocspruner_flops_33%.pth" --output-dir "eval_result_cifar"

# Vgg16-CIFAR10
python main_cifar.py --mode eval --model vgg16 --dataset "cifar10" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar10_vgg16_ocspruner_flops_20%.pth" --output-dir "eval_result_cifar"
python main_cifar.py --mode eval --model vgg16 --dataset "cifar10" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar10_vgg16_ocspruner_flops_26%.pth" --output-dir "eval_result_cifar"

# Vgg19-CIFAR100
python main_cifar.py --mode eval --model vgg19 --dataset "cifar100" --model-path "ocspruner_pruned_models/CIFAR-10-100/cifar100_vgg19_ocspruner_flops_11%.pth" --output-dir "eval_result_cifar"