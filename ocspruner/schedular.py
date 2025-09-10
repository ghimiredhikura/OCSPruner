
import torch
from engine.utils.imagenet_utils import utils

def define_optimizer_scheduler(model, trainLoader, args):
    learning_rate = args.learning_rate
    total_epochs = args.total_epochs
    total_warmup_epochs = args.total_warmup_epochs    

    weight_decay = args.weight_decay
    momentum = args.momentum

    """Defines the optimizer and the learning rate scheduler."""
    # Learning rate scheduler in the form (type, kwargs)
    tupleStr = learning_rate.strip()
    # Remove parenthesis
    if tupleStr[0] == '(':
        tupleStr = tupleStr[1:]
    if tupleStr[-1] == ')':
        tupleStr = tupleStr[:-1]
    name, *kwargs = tupleStr.split(',')
    if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
       scheduler = (name, kwargs)
       initial_lr = float(kwargs[0])
    else:
        raise NotImplementedError(f"LR Scheduler {scheduler} not implemented.")

    # Define the optimizer
    wd = weight_decay or 0.
    optimizer = torch.optim.SGD(params=model.parameters(), lr=initial_lr,
                                        momentum=momentum,
                                        weight_decay=wd, nesterov=wd > 0.)

    # We define a scheduler. All schedulers work on a per-iteration basis
    iterations_per_epoch = len(trainLoader)
    n_total_iterations = iterations_per_epoch * total_epochs
    n_warmup_iterations = 0

    # Set the initial learning rate
    for param_group in optimizer.param_groups: param_group['lr'] = initial_lr

    # Define the warmup scheduler if needed
    warmup_scheduler, milestone = None, None
    if total_warmup_epochs and total_warmup_epochs > 0:
        assert int(
            total_warmup_epochs) == total_warmup_epochs, "At the moment no float warmup allowed."
        n_warmup_iterations = int(float(total_warmup_epochs) * iterations_per_epoch)
        # As a start factor we use 1e-20, to avoid division by zero when putting 0.
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                                start_factor=1e-20, end_factor=1.,
                                                                total_iters=n_warmup_iterations)
        milestone = n_warmup_iterations + 1

    n_remaining_iterations = n_total_iterations - n_warmup_iterations

    name, kwargs = scheduler
    scheduler = None
    if name == 'Constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer,
                                                        factor=1.0,
                                                        total_iters=n_remaining_iterations)
    elif name == 'StepLR':
        # Tuple of form ('StepLR', initial_lr, step_size, gamma)
        # Reduces initial_lr by gamma every step_size epochs
        step_size, gamma = float(kwargs[1]), float(kwargs[2])

        # Convert to iterations
        step_size = round(iterations_per_epoch * step_size * total_epochs)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size,
                                                    gamma=gamma)
        
    elif name == 'MultiStepLR':
        # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
        # Reduces initial_lr by gamma every epoch that is in the list milestones
        milestones, gamma = kwargs[1].strip(), float(kwargs[2])
        # Remove square bracket
        if milestones[0] == '[':
            milestones = milestones[1:]
        if milestones[-1] == ']':
            milestones = milestones[:-1]

        # Convert to iterations directly
        milestones = [round(float(ms) * total_epochs) * iterations_per_epoch for ms in milestones.split('|')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                            gamma=gamma)
    elif name == 'ExponentialLR':
        # Tuple of form ('ExponentialLR', initial_lr, gamma)
        gamma = float(kwargs[1])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif name == 'Linear':
        # Tuple of form ('Linear', initial_lr)
        if len(kwargs) == 2:
            # The final learning rate has also been passed
            end_factor = float(kwargs[1]) / float(kwargs[0])
        else:
            end_factor = 0.
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                        start_factor=1.0, end_factor=end_factor,
                                                        total_iters=n_remaining_iterations)
    elif name == 'Cosine':
        # Tuple of form ('Cosine', initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=n_remaining_iterations, eta_min=0.)

    # Reset base lrs to make this work
    scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in optimizer.param_groups]

    # Define the Sequential Scheduler
    if warmup_scheduler is None:
        scheduler_final = scheduler
    elif name in ['StepLR', 'MultiStepLR']:
        # We need parallel schedulers, since the steps should be counted during warmup
        scheduler_final = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
    else:
        scheduler_final = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, scheduler],
                                                milestones=[milestone])
    return optimizer, scheduler_final

def define_optimizer_scheduler_imagenet(model, trainLoader, args, regularizer=None):
    learning_rate = args.learning_rate
    total_epochs = args.total_epochs
    total_warmup_epochs = args.total_warmup_epochs    

    weight_decay = args.weight_decay
    momentum = args.momentum

    """Defines the optimizer and the learning rate scheduler."""
    # Learning rate scheduler in the form (type, kwargs)
    tupleStr = learning_rate.strip()
    # Remove parenthesis
    if tupleStr[0] == '(':
        tupleStr = tupleStr[1:]
    if tupleStr[-1] == ')':
        tupleStr = tupleStr[:-1]
    name, *kwargs = tupleStr.split(',')
    if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
       scheduler = (name, kwargs)
       initial_lr = float(kwargs[0])
    else:
        raise NotImplementedError(f"LR Scheduler {scheduler} not implemented.")

    weight_decay = args.weight_decay #if regularizer is None else 0
    bias_weight_decay = args.bias_weight_decay #if regularizer is None else 0
    norm_weight_decay = args.norm_weight_decay #if regularizer is None else 0

    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))

    parameters = utils.set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=initial_lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=initial_lr, momentum=args.momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=initial_lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")


    # We define a scheduler. All schedulers work on a per-iteration basis
    iterations_per_epoch = len(trainLoader)
    n_total_iterations = iterations_per_epoch * total_epochs
    n_warmup_iterations = 0

    # Set the initial learning rate
    for param_group in optimizer.param_groups: param_group['lr'] = initial_lr

    # Define the warmup scheduler if needed
    warmup_scheduler, milestone = None, None
    if total_warmup_epochs and total_warmup_epochs > 0:
        assert int(
            total_warmup_epochs) == total_warmup_epochs, "At the moment no float warmup allowed."
        n_warmup_iterations = int(float(total_warmup_epochs) * iterations_per_epoch)
        # As a start factor we use 1e-20, to avoid division by zero when putting 0.
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                                start_factor=1e-20, end_factor=1.,
                                                                total_iters=n_warmup_iterations)
        milestone = n_warmup_iterations + 1

    n_remaining_iterations = n_total_iterations - n_warmup_iterations

    name, kwargs = scheduler
    scheduler = None
    if name == 'Constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer,
                                                        factor=1.0,
                                                        total_iters=n_remaining_iterations)
    elif name == 'StepLR':
        # Tuple of form ('StepLR', initial_lr, step_size, gamma)
        # Reduces initial_lr by gamma every step_size epochs
        step_size, gamma = float(kwargs[1]), float(kwargs[2])

        # Convert to iterations
        step_size = round(iterations_per_epoch * step_size * total_epochs)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size,
                                                    gamma=gamma)
        
    elif name == 'MultiStepLR':
        # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
        # Reduces initial_lr by gamma every epoch that is in the list milestones
        milestones, gamma = kwargs[1].strip(), float(kwargs[2])
        # Remove square bracket
        if milestones[0] == '[':
            milestones = milestones[1:]
        if milestones[-1] == ']':
            milestones = milestones[:-1]

        # Convert to iterations directly
        milestones = [round(float(ms) * total_epochs) * iterations_per_epoch for ms in milestones.split('|')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                            gamma=gamma)
    elif name == 'ExponentialLR':
        # Tuple of form ('ExponentialLR', initial_lr, gamma)
        gamma = float(kwargs[1])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif name == 'Linear':
        # Tuple of form ('Linear', initial_lr)
        if len(kwargs) == 2:
            # The final learning rate has also been passed
            end_factor = float(kwargs[1]) / float(kwargs[0])
        else:
            end_factor = 0.
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                        start_factor=1.0, end_factor=end_factor,
                                                        total_iters=n_remaining_iterations)
    elif name == 'Cosine':
        # Tuple of form ('Cosine', initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=n_remaining_iterations, eta_min=0.)

    # Reset base lrs to make this work
    scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in optimizer.param_groups]

    # Define the Sequential Scheduler
    if warmup_scheduler is None:
        scheduler_final = scheduler
    elif name in ['StepLR', 'MultiStepLR']:
        # We need parallel schedulers, since the steps should be counted during warmup
        scheduler_final = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
    else:
        scheduler_final = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, scheduler],
                                                milestones=[milestone])
    return optimizer, scheduler_final