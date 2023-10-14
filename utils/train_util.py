import torch
import random


def find_substring_in_list(substring, str_list):
    string_list = [name for name in str_list if name.startswith(substring)]
    return string_list


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def check_require_grad(model):
    require_grad = False
    for p in model.parameters():
        require_grad = require_grad | p.requires_grad
        if require_grad:
            break
    return require_grad


def check_models_require_grad(models, ignore_keys=[]):
    require_grad = False
    for key in models:
        if key in ignore_keys:
            continue
        model = models[key]
        require_grad = require_grad | check_require_grad(model)
        if require_grad:
            break
    return require_grad


class NetworkQueryCache:
    QUERY_QUEUE = []
    MAX_QUERY_SIZE = 50
    SUFFICIENT_QUERY_SIZE = 25

    def __init__(self):
        pass

    @staticmethod
    def update_cache(res):
        backup_res = {}
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                backup_res[k] = v.detach().cpu().clone()
        NetworkQueryCache.QUERY_QUEUE.append(backup_res)
        if len(NetworkQueryCache.QUERY_QUEUE) > NetworkQueryCache.MAX_QUERY_SIZE:
            NetworkQueryCache.QUERY_QUEUE.pop(0)

    @staticmethod
    def query_cache(idx=-1, use_random=True):
        if len(NetworkQueryCache.QUERY_QUEUE) < NetworkQueryCache.SUFFICIENT_QUERY_SIZE:
            return None
        if use_random:
            ch = random.choice(NetworkQueryCache.QUERY_QUEUE)
        else:
            ch = NetworkQueryCache.QUERY_QUEUE[idx]
        ret = {}
        for k, v in ch.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.cuda()
        return ret


class LinearSchedule:
    """Linearly scaled scheduler."""

    def __init__(self, initial_value, final_value, num_steps):
        self.initial_value = torch.as_tensor(initial_value)
        self.final_value = torch.as_tensor(final_value)
        self.num_steps = torch.as_tensor(num_steps)

    def get(self, step):
        """Get the value for the given step."""
        if self.num_steps == 0:
            return torch.full_like(step, self.final_value, dtype=torch.float32)
        alpha = torch.minimum(step / self.num_steps, torch.as_tensor(1.0))
        return (1.0 - alpha) * self.initial_value + alpha * self.final_value
