# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel.utils import is_module_wrapper


@HOOKS.register_module()
class IterBasedEMAHook(Hook):
    def __init__(self,
                 momentum=0.0002,
                 ema_interval=1,
                 skip_iters=1000,
                 eval_interval=1000):
        assert isinstance(ema_interval, int) and ema_interval > 0
        assert 0 < momentum < 1

        self.skip_iters = skip_iters
        self.ema_interval = ema_interval
        self.eval_interval = eval_interval

        self.buffer_init_mode = True
        self.eval_mode = False
        self.param_ema_buffer = {}
        self.model_parameters = {}
        self.model_buffers = {}
        self.momentum = momentum ** ema_interval

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            buffer_name = f"ema_{name.replace('.', '_')}"  # "." is not allowed in module's buffer name

            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))

    def before_train_iter(self, runner):
        if self.eval_mode:
            self._swap_ema_parameters()
            self.eval_mode = False

    def after_train_iter(self, runner):
        curr_iter = runner.iter + 1
        if curr_iter <= self.skip_iters:
            return

        if curr_iter % self.ema_interval == 0:
            for name, parameter in self.model_parameters.items():
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]

                if self.buffer_init_mode:
                    buffer_parameter.copy_(parameter.data)
                else:
                    buffer_parameter.mul_(1.0 - self.momentum).add_(parameter.data, alpha=self.momentum)

            self.buffer_init_mode = False

        if curr_iter > 1 and curr_iter % self.eval_interval == 0:
            assert not self.eval_mode

            self._swap_ema_parameters()
            self.eval_mode = True

    def _swap_ema_parameters(self):
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
