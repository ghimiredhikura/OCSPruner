import os, argparse
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import vainf_torch_pruning.torch_pruning as tp
import engine.registry as registry
import engine.utils.utils as utils_cifar

from ocspruner.schedular import *
from ocspruner.pruner import get_pruner
from ocspruner.cost import calculate_channel_cost_ingroup
from ocspruner.utils import *

parser = argparse.ArgumentParser(description="One-cycle Structured Pruning with Stability Driven Structure Search (OCSPruner)", add_help=True)

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["baseline", "prune", "eval"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="cifar100", choices=['cifar10', 'cifar100'])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--total-warmup-epochs", type=int, default=0)
parser.add_argument("--total-epochs", type=int, default=300)
parser.add_argument("--learning-rate", default='(MultiStepLR, 0.1, [0.3|0.6|0.8], 0.2)', type=str, help="learning rate")
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument('--output-dir', default='result', help='path where to save')
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9, help='Momentum for MomentumOptimizer. default:0.9')
parser.add_argument("--seed", type=int, default=None)

# Pruning options
parser.add_argument("--method", type=str, default="group_norm_sl_ocspruner")
parser.add_argument("--target-flop-rr", type=float, default=0.6)
parser.add_argument("--reg", type=float, default=1e-4, help="growing regularization factor")
parser.add_argument("--reg-delta", type=float, default=1e-4, help="growing regularization increment factor")
parser.add_argument("--pruning-stability-thresh", type=float, default=0.999)
parser.add_argument("--sl-start-epoch", type=int, default=30, help="epochs for sparsity learning")
parser.add_argument("--sl-end-epoch", type=int, default=130, help="end sparsity learning upper limit")
parser.add_argument("--reg-add-interval", type=int, default=1, help="add reg_delta in each reg_add_interval epoch")
parser.add_argument("--prune-monitor-start-epoch", type=int, default=5, help="start pruned net monitoring")
parser.add_argument("--layer-prune-limit", type=float, default=0.75, help="max pruning limit for each layer")
parser.add_argument("--layer-dynamic-prune-limit-fact", type=float, default=1.0, help="dynamically adjust layer max pruning limit in each layer")

args = parser.parse_args()
torch.set_float32_matmul_precision('high')

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum").item()
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    avg_loss = loss / total
    return accuracy, avg_loss

def train_epoch(
    model,
    train_loader,
    optimizer, 
    scheduler,
    epochs,
    epoch,
    pruner=None,
    sl_start_epoch=1,
    prune_monitor_start_epoch=0,
):
    sl_regularize = True if (pruner is not None and epoch > sl_start_epoch+1) else False

    # Draw pruning group and nonpruning group hist
    if pruner is not None and epoch > prune_monitor_start_epoch:
        out_dir = os.path.join(args.output_dir, 'plot_group_norm')
        os.makedirs(out_dir, exist_ok=True)
        save_file = os.path.join(out_dir, f'hist{epoch}.jpg')
        pruner.plot_group_norm_hist(pruner.groups_prune, pruner.groups_noprune, save_file)
        #pruner.plot_all_group_norms()

    if sl_regularize and epoch % args.reg_add_interval == 0:
        args.reg += args.reg_delta

    # train
    device = args.device
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)

        if sl_regularize:
            l2_reg = pruner.get_prune_groups_total_norm(p=2) * args.reg
            loss += l2_reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        if sl_regularize: # for sparsity learning
            pruner.scale_weights_towards_zero(weight_scale_fact = (1 - args.reg * optimizer.param_groups[0]['lr']))

        if i % 10 == 0 and args.verbose:
            args.logger.info(
                "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                    epoch,
                    epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )

def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer, 
    scheduler,
    start_epoch=0,
    save_as=None,
    save_state_dict_only=True,
    pruner=None,
    sl_start_epoch=0,
    prune_monitor_start_epoch=1,
    method="",
    ori_ops=None, 
    ori_size=None,
    stable_pruning_epoch=0,
):
    device = args.device
    model.to(device)
    best_acc = -1
    running_sim_avg = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    last_epoch = 0

    if pruner is not None:
        out_dir = os.path.join(args.output_dir, 'plot_pruned_structure')
        os.makedirs(out_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        # Step 1: Update pruning groups until network stabalizes
        if pruner is not None and epoch >= prune_monitor_start_epoch and running_sim_avg <= args.pruning_stability_thresh:
            groups_pruning_indices, model_pruned = get_groups_pruning_indices(pruner, args)
            pruner.update_pruning_groups(groups_pruning_indices, need_noprune_group_too=True)
            save_file = os.path.join(out_dir, f'net_pruned{epoch}.pdf')
            plot_prune_retained_filter_ratios(model, model_pruned, save_file)
        if pruner is not None:
            sim_val = 0.0 if epoch < prune_monitor_start_epoch else get_net_similarity(pruner)
            running_sim_avg = draw_pi_sim(sim_val, save_as.replace(".pth", ".pdf"))
            args.logger.info("Pruning Stability Indicator = %.4f, reg = %0.4f" % (running_sim_avg, args.reg))

        # prune model after stabalize and check the val accuracy.        
        #if running_sim_avg > args.pruning_stability_thresh and epoch % 5 == 0:
        #    model_pruned = deepcopy(model)
        #    pruner_tmp = get_pruner(model_pruned, args, pruner.example_inputs)
        #    model_pruned = prune_model(pruner_tmp, test_loader, args, pruner.groups_pruning_indices, ori_ops, ori_size, True)
        
        # Train 
        model.train()
        train_epoch(model, train_loader, optimizer, scheduler, epochs, epoch, pruner, sl_start_epoch, prune_monitor_start_epoch)
        last_epoch = epoch

        # Eval
        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info("Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]))

        # Save best model
        if best_acc < acc:
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}_{}.pth".format(args.dataset, args.model, args.method) )
                    if epoch == sl_start_epoch:
                        save_as = os.path.join( args.output_dir, "{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, epoch))
                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "baseline":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}.pth".format(args.dataset, args.model) )
                torch.save(model.state_dict(), save_as)

            best_acc = acc

        if pruner is not None and (epoch > args.sl_end_epoch or running_sim_avg >= args.pruning_stability_thresh):
            break

    args.logger.info("Best Acc=%.4f" % (best_acc))
    return last_epoch

def get_groups_pruning_indices(pruner, args, base_ops=None, base_size=None):
    if base_ops is None and base_size is None:
        base_ops, base_size = pruner.get_model_info()
    all_groups_with_imp = pruner.get_all_group_imp_scores(layer_max_prune_limit=args.layer_prune_limit, 
                                            dynamic_pruning_limit_fact=args.layer_dynamic_prune_limit_fact)
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
        if abs(args.target_flop_rr - flop_rr) <= 5e-4:
            break
        if flop_rr <= args.target_flop_rr:
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

def prune_model(pruner, test_loader, args, group_pruning_indices=None, base_ops=None, base_size=None, min_log=False):
    # Pruning
    model = pruner.model
    if not min_log:
        args.logger.info(model)
    model.eval()
    if base_ops is None and base_size is None:
        base_ops, base_size = pruner.get_model_info()
    #if not min_log:
    ori_acc, ori_val_loss = eval(model, test_loader, device=args.device)
    args.logger.info("Pruning...")
    pruner.pruning(group_pruning_indices)
    if not min_log:
        args.logger.info(model)
    pruned_ops, pruned_size = pruner.get_model_info()
    #if not min_log:
    pruned_acc, pruned_val_loss = eval(model, test_loader, device=args.device)
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
    args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc) )
    args.logger.info(
        "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss) )
    return model

def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Logger
    if args.mode == "baseline":
        logger_name = "{}-{}".format(args.dataset, args.model)
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "prune":
        logger_name = "{}-{}-{}-pruning-rate-{}".format(args.dataset, args.method, args.model, args.target_flop_rr)
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode, logger_name)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "eval":
        filename = os.path.basename(args.model_path)
        logger_name = os.path.splitext(filename)[0]
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    args.logger = utils_cifar.get_logger(logger_name, output=log_file)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, train_dst, val_dst, input_size = registry.get_dataset(args.dataset, data_root="data")
    args.num_classes = num_classes
    model = registry.get_model(args.model, num_classes=num_classes, pretrained=False, target_dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, num_workers=4, drop_last=True, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dst, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    if args.mode == "prune" or args.mode == "baseline":
        for k, v in utils_cifar.flatten_dict(vars(args)).items():  # print args
            args.logger.info("%s: %s" % (k, v))
    
    model = model.to(args.device)

    ######################################################
    example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    if args.mode == "baseline":
        ops, params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs,)
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))

        # get optimizer and regulizer 
        optimizer, scheduler = define_optimizer_scheduler(model, train_loader, args)

        start_time = time.time()
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.total_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        hours, rest = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rest, 60)
        args.logger.info(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    elif args.mode == "prune":

        start_time = time.time()

        # pruner
        cost = calculate_channel_cost_ingroup(model, args, example_inputs)
        pruner = get_pruner(model, args, example_inputs)

        pruner.group_cost = cost
        ori_ops, ori_size = pruner.get_model_info()
        # get optimizer and schedular
        sl_optimizer, sl_scheduler = define_optimizer_scheduler(model, train_loader, args)

        net_pth = "{}_{}_{}.pth".format(args.dataset, args.model, args.method)
        net_pth = os.path.join( os.path.join(args.output_dir, net_pth) )
        
        # Step 1. Training with Sparsity Leanring 
        args.logger.info("Step 1: Training with Sparsity Learning...")
        last_epoch = train_model(
            model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.total_epochs,
            optimizer=sl_optimizer,
            scheduler=sl_scheduler,
            pruner=pruner,
            save_state_dict_only=True,
            save_as=net_pth,
            sl_start_epoch=args.sl_start_epoch,
            prune_monitor_start_epoch=args.prune_monitor_start_epoch,
            method=args.method,
            ori_ops=ori_ops,
            ori_size=ori_size,
        )

        # Step 2. Pruning
        args.logger.info("Step 2: Pruning...")
        pruned_model = prune_model(pruner, test_loader, args, pruner.groups_pruning_indices, ori_ops, ori_size)
        del pruner # delete pruner 

        # Step 3. Training pruned model...
        args.logger.info("Step 3: Training Pruned Model...")
        # get optimizer and schedular for pruned net
        optimizer, scheduler = define_optimizer_scheduler(pruned_model, train_loader, args)
        for _ in range(last_epoch):
            for _ in range(len(train_loader)):
                scheduler.step()

        train_model(
            pruned_model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.total_epochs,
            start_epoch=last_epoch+1,
            optimizer=optimizer,
            scheduler=scheduler,
            save_state_dict_only=False
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        hours, rest = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rest, 60)
        args.logger.info(f"ESSP-GLS Elapsed Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    elif args.mode == "eval":
        model.eval()
        base_ops, base_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)

        if args.model_path is not None:
            args.logger.info("Loading model from {restore}".format(restore=args.model_path) )
            model = torch.load(args.model_path, map_location="cpu")
            model = model.to(args.device)
            args.logger.info("{}".format(model))

        model.eval()
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        acc, val_loss = eval(model, test_loader, device=args.device)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}".format(acc, val_loss))
        if base_ops == pruned_ops:
            args.logger.info("Params: {:.2f} M".format(base_size / 1e6))
            args.logger.info("ops: {:.2f} M".format(base_ops / 1e6))
        else:
            args.logger.info(
                "Params: {:.2f} M => {:.2f} M, (Param RR {:.2f}%)".format(
                    base_size / 1e6, pruned_size / 1e6, (1.0 - pruned_size / base_size) * 100 ))
            args.logger.info(
                "FLOPs: {:.2f} M => {:.2f} M (FLOPs RR {:.2f}%, Speed-Up {:.2f}X )".format(
                    base_ops / 1e6,
                    pruned_ops / 1e6,
                    (1.0 - pruned_ops / base_ops) * 100,
                    base_ops / pruned_ops ))

if __name__ == "__main__":
    main()