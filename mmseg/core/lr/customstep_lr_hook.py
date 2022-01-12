# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.hooks import HOOKS

from .base_lr_hook import BaseLrUpdaterHook


@HOOKS.register_module()
class CustomstepLrUpdaterHook(BaseLrUpdaterHook):
    def __init__(self, step, gamma=0.1, **kwargs):
        super(CustomstepLrUpdaterHook, self).__init__(**kwargs)

        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s >= 0
        elif isinstance(step, int):
            assert step >= 0
        else:
            raise TypeError('"step" must be a list or integer')

        self.steps = step if isinstance(step, (tuple, list)) else [step]
        self.gamma = gamma

    def _init_states(self, runner):
        super(CustomstepLrUpdaterHook, self)._init_states(runner)

        if self.by_epoch:
            self.steps = [
                step * self.epoch_len for step in self.steps
            ]

    def get_lr(self, runner, base_lr):
        progress = runner.iter

        skip_iters = self.fixed_iters + self.warmup_iters
        if progress <= skip_iters:
            return base_lr

        exp = len(self.steps)
        for i, s in enumerate(self.steps):
            if progress < s:
                exp = i
                break

        return base_lr * self.gamma**exp
