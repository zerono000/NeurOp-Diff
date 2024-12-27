import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomIterationScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_sequence, step_size, last_iteration=-1):
        self.lr_sequence = lr_sequence
        self.step_size = step_size
        self.current_step = last_iteration
        self.lr_index = 0
        super(CustomIterationScheduler, self).__init__(optimizer, last_epoch=last_iteration)

    def get_lr(self):
        if self.lr_index < len(self.lr_sequence):
            return [self.lr_sequence[self.lr_index] for _ in self.optimizer.param_groups]
        else:
            return [self.lr_sequence[-1] for _ in self.optimizer.param_groups]

    def step(self, iteration=None):
        if iteration is None:
            self.current_step += 1
        else:
            self.current_step = iteration

        new_lr_index = self.current_step // self.step_size

        if new_lr_index > self.lr_index and self.lr_index < len(self.lr_sequence) - 1:
            self.lr_index = new_lr_index
            super(CustomIterationScheduler, self).step()
