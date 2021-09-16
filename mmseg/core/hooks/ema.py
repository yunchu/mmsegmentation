from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel.utils import is_module_wrapper


@HOOKS.register_module()
class IterBasedEMAHook(Hook):
    def __init__(self,
                 momentum=0.0002,
                 ema_interval=1,
                 warm_up=100,
                 eval_interval=1000):
        assert isinstance(ema_interval, int) and ema_interval > 0
        assert 0 < momentum < 1

        self.warm_up = warm_up
        self.ema_interval = ema_interval
        self.eval_interval = eval_interval

        self.eval_mode = False
        self.param_ema_buffer = None
        self.model_parameters = None
        self.model_buffers = None
        self.momentum = momentum ** ema_interval

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))

    def before_train_iter(self, runner):
        if self.eval_mode:
            self._swap_ema_parameters()
            self.eval_mode = False

    def after_train_iter(self, runner):
        curr_step = runner.iter

        if curr_step % self.ema_interval == 0:
            momentum = min(self.momentum, (1 + curr_step) / (self.warm_up + curr_step))

            for name, parameter in self.model_parameters.items():
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(parameter.data, alpha=momentum)

        if curr_step > 0 and (curr_step + 1) % self.eval_interval == 0:
            assert not self.eval_mode

            self._swap_ema_parameters()
            self.eval_mode = True

    def after_run(self, runner):
        if not self.eval_mode:
            self._swap_ema_parameters()
            self.eval_mode = True

    def _swap_ema_parameters(self):
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
