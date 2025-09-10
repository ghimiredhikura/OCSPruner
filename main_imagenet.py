import datetime
import os, sys, platform
import time
import warnings

import engine.registry as registry
import vainf_torch_pruning.torch_pruning as tp

from copy import deepcopy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from engine.utils.imagenet_utils import presets, transforms, utils, sampler
from engine.utils.utils import get_logger

import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data.dataloader import default_collate

from ocspruner.schedular import *
from ocspruner.pruner import get_pruner
from ocspruner.cost import calculate_channel_cost_ingroup
from ocspruner.utils import *

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="One-cycle Structured Pruning with Stability Driven Structure Search (OCSPruner)", add_help=add_help)

    parser.add_argument("--data-path", default="D:/ImageNet/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--total-epochs", type=int, default=120)
    parser.add_argument("--total-warmup-epochs", type=int, default=0)
    parser.add_argument("--learning-rate", default='(Linear, 0.01)', type=str, help="learning rate")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    
    # pruning parameters
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--train-resume", action="store_true")
    parser.add_argument("--method", type=str, default='l1')
    parser.add_argument("--global-pruning", default=False, action="store_true")
    parser.add_argument("--target-flops-rr", type=float, default=0.6)
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--reg-delta", type=float, default=1e-3)
    parser.add_argument("--reg-add-interval", type=int, default=1, help="add reg_delta in each reg_add_interval epoch")

    parser.add_argument("--pruning-stability-thresh", type=float, default=0.98)
    parser.add_argument("--sl-resume", type=str, default=None)
    parser.add_argument("--sl-start-epoch", type=int, default=15, help="epochs for sparsity learning")
    parser.add_argument("--sl-end-epoch", type=int, default=36, help="epochs for sparsity learning")
    parser.add_argument("--prune-monitor-start-epoch", type=int, default=5, help="start pruned net monitoring")
    parser.add_argument("--layer-prune-limit", type=float, default=0.75)

    return parser

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", details=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    args.logger.info(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    if not details:
        return metric_logger.acc1.global_avg
    else:
        return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, args):
    # Data loading code
    args.logger.info("Loading data...")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)

    args.logger.info("Loading training data...")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        args.logger.info("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                              random_erase_prob=random_erase_prob))
        if args.cache_dataset:
            args.logger.info("Saving dataset_train to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    args.logger.info("Data loading took {}".format (time.time() - st))

    args.logger.info("Loading validation data...")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        args.logger.info("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))
        if args.cache_dataset:
            args.logger.info("Saving dataset_test to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    args.logger.info("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def get_groups_pruning_indices(pruner, args, base_ops=None, base_size=None):
    if base_ops is None and base_size is None:
        base_ops, base_size = pruner.get_model_info()
    all_groups_with_imp = pruner.get_all_group_imp_scores(layer_max_prune_limit=args.layer_prune_limit)
    all_imp = torch.cat([local_imp[-1] for local_imp in all_groups_with_imp], dim=0)
    all_imp_sorted, sorted_indices = torch.sort(all_imp)
    # Find indices where the value is not 100
    not_100_indices = torch.nonzero(all_imp_sorted != 100.0).squeeze()
    # Get sorted indices excluding the 100s
    sorted_indices_without_100s = sorted_indices[not_100_indices]
    args.logger.info("Pruning to Target FLOPs [Binary Search]...")
    # Initialize search space
    left, right = 0, len(sorted_indices_without_100s) - 1
    # Loop until the search space is exhausted
    while left <= right:
        # Calculate the midpoint index
        idx = (left + right) // 2
        model_pruned = deepcopy(pruner.model)
        pruner_tmp = get_pruner(model_pruned, args, pruner.example_inputs)
        groups_pruning_indices = pruner.get_group_pruning_indices(all_groups_with_imp, all_imp, sorted_indices_without_100s[idx])
        pruner_tmp.pruning(groups_pruning_indices)
        pruned_ops, pruned_size = pruner_tmp.get_model_info()
        flop_rr = 1.0 - float(pruned_ops) / base_ops
        if abs(args.target_flops_rr - flop_rr) <= 5e-4:
            break
        if flop_rr <= args.target_flops_rr:
            left = idx + 1
        else:
            right = idx - 1
    args.logger.info(
        "Params: {:.2f} M => {:.2f} M, (Param RR {:.2f}%)".format(
            base_size / 1e6, pruned_size / 1e6, (1.0 - pruned_size / base_size) * 100 ))
    args.logger.info(
        "FLOPs: {:.2f} M => {:.2f} M (FLOPs RR {:.2f}%, Speed-Up {:.2f}X )".format(
            base_ops / 1e6,
            pruned_ops / 1e6,
            (1.0 - pruned_ops / base_ops) * 100,
            base_ops / pruned_ops ))
    return groups_pruning_indices, model_pruned

def prune_model(pruner, test_loader, args, criteria, group_pruning_indices=None, base_ops=None, base_size=None, min_log=False):
    # Pruning
    model = pruner.model
    if not min_log:
        args.logger.info(model)
    model.eval()
    if base_ops is None and base_size is None:
        base_ops, base_size = pruner.get_model_info()
    #if not min_log:
    ori_acc1, ori_acc5 = evaluate(model, criteria, test_loader, device=args.device, details=True)
    args.logger.info("Pruning...")
    pruner.pruning(group_pruning_indices)
    if not min_log:
        args.logger.info(model)
    pruned_ops, pruned_size = pruner.get_model_info()
    #if not min_log:
    pruned_acc1, pruned_acc5 = evaluate(model, criteria, test_loader, device=args.device, details=True)
    args.logger.info(
            "Params: {:.2f} M => {:.2f} M, (Param RR {:.2f}%)".format(
                base_size / 1e6, pruned_size / 1e6, (1.0 - pruned_size / base_size) * 100 ))
    args.logger.info(
            "FLOPs: {:.2f} M => {:.2f} M (FLOPs RR {:.2f}%, Speed-Up {:.2f}X )".format(
                base_ops / 1e6,
                pruned_ops / 1e6,
                (1.0 - pruned_ops / base_ops) * 100,
                base_ops / pruned_ops ))
    #if not min_log:
    args.logger.info("Acc1: {:.4f} => {:.4f}, Acc5: {:.4f} => {:.4f}".format(ori_acc1, pruned_acc1, ori_acc5, pruned_acc5) )
    return model

def train_one_epoch(model, criterion, optimizer, scheduler, data_loader, 
                    device, epoch, args, sl_start_epoch=False, prune_monitor_start_epoch=0, method="",
                    model_ema=None, scaler=None, pruner=None, recover=None,):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    sl_regularize = True if (pruner is not None and sl_start_epoch) else False

    # Draw pruning group and nonpruning group hist
    if pruner is not None and epoch > prune_monitor_start_epoch:
        out_dir = os.path.join(args.output_dir, 'plot_group_norm')
        os.makedirs(out_dir, exist_ok=True)
        save_file = os.path.join(out_dir, f'hist{epoch}.jpg')
        pruner.plot_group_norm_hist(pruner.groups_prune, pruner.groups_noprune, save_file)
        save_file = os.path.join(out_dir, f'hist{epoch}.jpg')
        pruner.plot_all_group_norms()

    if sl_regularize and epoch % args.reg_add_interval == 0:
        args.reg += args.reg_delta

    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        if sl_regularize:
            l2_reg = pruner.get_prune_groups_total_norm() * args.reg
            loss += l2_reg

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if sl_regularize:
                scaler.unscale_(optimizer)
                #regularizer(model)
                pruner.scale_weights_towards_zero(weight_scale_fact = (1 - args.reg * optimizer.param_groups[0]['lr']))
            if recover:
                recover(model.module)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if sl_regularize:
                #regularizer(model)
                pruner.scale_weights_towards_zero(weight_scale_fact = (1 - args.reg * optimizer.param_groups[0]['lr']))
            if recover:
                recover(model.module)
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        scheduler.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def train(
    model, 
    epochs,
    optimizer, 
    lr_scheduler,
    criterion, 
    train_sampler, data_loader, data_loader_test, 
    device, args, pruner=None, state_dict_only=True, recover=None,
    sl_start_epoch_man=0, prune_monitor_start_epoch=1, method="",
    ori_ops=None, ori_size=None,):

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None    
    args.logger.info("Mixed Precision: {}".format(args.amp))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    '''
    if args.resume:
        args.logger.info("resume model: ", args.resume)
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp = checkpoint["model"]
        #if not args.test_only:
        #    optimizer.load_state_dict(checkpoint["optimizer"])
        #    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] - 15
        for _ in range(args.start_epoch):
            for _ in range(len(data_loader)):
                lr_scheduler.step()
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    '''
    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return
    
    if pruner is not None:
        out_dir = os.path.join(args.output_dir, 'plot_pruned_structure')
        os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()
    best_acc = 0
    acc = 0
    prefix = '' if pruner is None else 'regularized_{:e}_'.format(args.reg)
    last_epoch = 0

    running_sim_avg = 0
    # Initialize an array to store the previous 5 values of running_sim_avg
    running_sim_avg_past_vals = []

    sl_start_epoch = False
    sl_start_epoch_no = -1

    for epoch in range(args.start_epoch, epochs):
        torch.cuda.empty_cache()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Step 1: Update pruning groups until the network stabilizes
        if pruner is not None and epoch >= prune_monitor_start_epoch and running_sim_avg <= args.pruning_stability_thresh:
            model.eval()
            groups_pruning_indices, model_pruned = get_groups_pruning_indices(pruner, args)
            pruner.update_pruning_groups(groups_pruning_indices, need_noprune_group_too=True)
            save_file = os.path.join(out_dir, f'net_pruned{epoch}.jpg')
            plot_prune_retained_filter_ratios(model, model_pruned, save_file)
        
        if pruner is not None:
            sim_val = 0.0 if epoch < prune_monitor_start_epoch else get_net_similarity(pruner)
            save_as = os.path.join(args.output_dir, 'psi_curve_net.jpg')
            running_sim_avg = draw_pi_sim(sim_val, save_as)
            args.logger.info("Pruning Stability Indicator (raw) = %.4f" % (sim_val))
            args.logger.info("Pruning Stability Indicator (avg) = %.4f, reg = %0.4f" % (running_sim_avg, args.reg))

            # Update the array with the current running_sim_avg
            running_sim_avg_past_vals.append(running_sim_avg)
            # Trim the array to keep only the last 3 values
            if len(running_sim_avg_past_vals) > 3:
                running_sim_avg_past_vals.pop(0)

            # Check if sl_start_epoch should be set to True
            if epoch > 3 and sl_start_epoch is False and (running_sim_avg - running_sim_avg_past_vals[-3]) <= 0.0001:
                sl_start_epoch = True
                sl_start_epoch_no = epoch
                args.logger.info("Automatic $t_{sl-start}$) = %d" % (epoch))

        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, args,
                        sl_start_epoch, prune_monitor_start_epoch, method,
                        model_ema, scaler, pruner, recover=recover)
        last_epoch = epoch
        # lr_scheduler.step()
        
        if epoch > 100 or epoch % 10 == 0:
            acc = evaluate(model, criterion, data_loader_test, device=device)
            if model_ema:
                acc = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict() if state_dict_only else model_without_ddp,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc > best_acc:
                best_acc = acc
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, prefix+"best.pth"))
            if epoch == sl_start_epoch_no or epoch == 0:
                file_name = f"checkpoint_{epoch}.pth"
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, file_name))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, prefix+"latest.pth"))
        
        args.logger.info("Epoch {}/{}, Current Best Acc = {:.6f}".format(epoch, epochs, best_acc))

        if pruner is not None and epoch >= prune_monitor_start_epoch:
            if epoch >= args.sl_end_epoch or running_sim_avg >= args.pruning_stability_thresh:
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    args.logger.info(f"Training time {total_time_str}")

    return last_epoch

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    logger_name = "{}-log".format(args.model, args.target_flops_rr)
    log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    args.logger = get_logger(logger_name, output=log_file)

    #args.distributed = False
    utils.init_distributed_mode(args)
    args.logger.info(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    args.logger.info("Creating model")
    args.num_classes = 1000
    model = registry.get_model(num_classes=args.num_classes, name=args.model, pretrained=False, target_dataset='imagenet') 
    model.eval()
    model.to(args.device)
    args.logger.info("="*116)
    #args.logger.info(model)
    example_inputs = torch.randn(1, 3, 224, 224).to(args.device)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    args.logger.info("Params: {:.4f} M".format(base_params / 1e6))
    args.logger.info("ops: {:.4f} G".format(base_ops / 1e9))
    args.logger.info("="*116)

    if args.label_smoothing>0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # ocspruner Training and Pruning 
    if args.prune:
        cost = calculate_channel_cost_ingroup(model, args, example_inputs)
        pruner = get_pruner(model, example_inputs=example_inputs, args=args)
        pruner.group_cost = cost
        ori_ops, ori_size = pruner.get_model_info()
        optimizer, lr_scheduler = define_optimizer_scheduler_imagenet(model, data_loader, args)

        if args.resume:
            args.logger.info("resume model: {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            
            if not args.test_only:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"]
        
        start_time = time.time()

        args.logger.info("Pruning Structure Search and Sparcity Learning...")
        last_epoch = train(model, args.total_epochs, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
            device=device, args=args, pruner=pruner, state_dict_only=True,
            sl_start_epoch_man=args.sl_start_epoch, prune_monitor_start_epoch=args.prune_monitor_start_epoch,
            method=args.method, ori_ops=ori_ops, ori_size=ori_size,
        )

        args.logger.info("Pruning model...")
        groups_pruning_indices, model_pruned = get_groups_pruning_indices(pruner, args)
        pruner.update_pruning_groups(groups_pruning_indices, need_noprune_group_too=True)
        pruned_model = prune_model(pruner, data_loader_test, args, criterion, pruner.groups_pruning_indices, ori_ops, ori_size)

        del pruner 
        args.logger.info("="*116)

        # Step 3. Training for remaining epochs after Pruning
        # get optimizer and regulizer for pruned net
        optimizer, scheduler = define_optimizer_scheduler_imagenet(pruned_model, data_loader, args)
        for epoch in range(last_epoch+1):
            for _ in range(len(data_loader)):
                scheduler.step()

        args.logger.info("Continue Training after Pruning ...")
        args.resume = ""
        args.start_epoch = last_epoch+1
        train(pruned_model, args.total_epochs, 
                optimizer=optimizer, lr_scheduler=scheduler, criterion=criterion,
                train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                device=device, args=args, pruner=None, state_dict_only=False)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        args.logger.info(f"Total (Pruning + Training) Time Elapsed: {total_time_str}")

    elif args.train_resume:
        dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
        if args.resume:
            args.logger.info("resume model: {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp = checkpoint["model"]
            print(model_without_ddp)

            optimizer, scheduler = define_optimizer_scheduler_imagenet(model_without_ddp, data_loader, args)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"]
            args.logger.info("resume last epoch: {}".format(checkpoint["epoch"]))
            model = model_without_ddp

            for _ in range(args.start_epoch):
                for _ in range(len(data_loader)):
                    scheduler.step()
        else: 
            optimizer, scheduler = define_optimizer_scheduler_imagenet(model, data_loader, args)

        start_time = time.time()

        train(model, args.total_epochs+1, 
                optimizer=optimizer, lr_scheduler=scheduler, criterion=criterion,
                train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                device=device, args=args, pruner=None, state_dict_only=False)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        args.logger.info(f"Total Pruning + Training Time Elapsed: {total_time_str}")
    
    elif args.eval:
        args.logger.info(args.model_path)
        # Load the model from the provided model path
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            # If the loaded checkpoint is a dictionary containing a model
            model_pruned = checkpoint["model"]
        else:
            # If the loaded checkpoint directly contains the model
            model_pruned = checkpoint
        args.logger.info(model_pruned)
        model_pruned.to(args.device)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model_pruned, example_inputs=example_inputs)
        args.logger.info("Params: {:.2f} M => {:.2f} M, (Param RR {:.2f}%)".format(
                base_params / 1e6, pruned_size / 1e6, (1.0 - pruned_size / base_params) * 100 ))
        args.logger.info("FLOPs: {:.2f} M => {:.2f} M (FLOPs RR {:.2f}%, Speed-Up {:.2f}X )".format(
                base_ops / 1e6,
                pruned_ops / 1e6,
                (1.0 - pruned_ops / base_ops) * 100,
                base_ops / pruned_ops ))
        evaluate(model_pruned, criterion, data_loader_test, device=args.device, details=True)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)