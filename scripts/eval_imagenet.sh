
# eval resnet18 pruned model
python main_imagenet.py --eval --model resnet18 --model-path "ocspruner_pruned_models\ImageNet\ResNet18_Prune_45%[FLOPs]\pruned_model.pth" --output-dir "eval_result_imagenet"

python main_imagenet.py --eval --model resnet50 --model-path "ocspruner_pruned_models\ImageNet\ResNet50_Prune_67%[FLOPs]\pruned_model.pth" --output-dir "eval_result_imagenet"