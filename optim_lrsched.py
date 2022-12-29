import torch.optim as optim
import math


def optims_and_scheds(param_groups, param_lr, has_sched, config):
    """
        All the arguments are lists.
        param_groups: List/tuple of parameter groups
        param_lr: List/tuple of learning rates for each param
        has_sched: List/tuple of boolean values, telling which parameter groups have the lr scheduler
        config: configuration variable
    """
    assert type(param_groups) == list or type(param_groups) == tuple
    assert type(param_lr) == list or type(param_lr) == tuple
    assert (type(has_sched) == list or type(
        has_sched) == tuple) and type(has_sched[0]) == bool
    assert len(param_groups) == len(has_sched) == len(
        param_lr), "Lenghts of arguments are not same"

    optimizers = []
    lr_schedulers = []

    for i in range(len(param_groups)):
        optimizers.append(optim.SGD(params=param_groups[i], lr=param_lr[i], momentum=0.9, nesterov=True))

    epochs = max(config["model"]["epochs"] - 1, 1) 
    gamma = math.pow((config["lr"]["cls_lr_end"] / config["lr"]["cls_lr_start"]), (1/epochs))

    for i in range(len(optimizers)):
        if has_sched[i]:
            lr_schedulers.append(optim.lr_scheduler.StepLR(optimizer=optimizers[i], step_size=1, gamma=gamma, verbose=True))
    print("There are {0} optimizers and {1} lr schedulers".format(len(optimizers), len(lr_schedulers)))
    return (optimizers, lr_schedulers)
