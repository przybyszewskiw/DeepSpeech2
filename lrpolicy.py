from enum import Enum


class LrPolicy(Enum):
    POLY_DECAY = 1,
    EXP_DECAY = 2,
    NO_DECAY = 3


def apply_policy(optimizer, policy_type, policy_params, optimizer_steps=None):
    if policy_type == LrPolicy.POLY_DECAY:
        return  # TODO
    elif policy_type == LrPolicy.EXP_DECAY:
        ds = max(policy_params["decay_steps"], 1)
        dr = policy_params["decay_rate"]
        min_lr = policy_params["min_lr"]

        if optimizer_steps % ds == ds - 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= dr
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr
