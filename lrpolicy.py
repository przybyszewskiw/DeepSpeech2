from enum import Enum


class LrPolicy(str, Enum):
    POLY_DECAY = "POLY_DECAY",
    EXP_DECAY = "EXP_DECAY",
    NO_DECAY = "NO_DECAY"


def apply_policy(optimizer, optimizer_steps, policy_type, policy_params):
    if policy_type == LrPolicy.POLY_DECAY:
        ds = max(policy_params["decay_steps"], 1)
        power = policy_params["power"]
        min_lr = max(policy_params["min_lr"], 0)
        max_iter = policy_params["max_iter"]
        init_lr = policy_params["lr"]

        if optimizer_steps % ds == ds - 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr * (1 - optimizer_steps / max_iter) ** power
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr

    elif policy_type == LrPolicy.EXP_DECAY:
        ds = max(policy_params["decay_steps"], 1)
        dr = policy_params["decay_rate"]
        min_lr = max(policy_params["min_lr"], 0)

        if optimizer_steps % ds == ds - 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= dr
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr
