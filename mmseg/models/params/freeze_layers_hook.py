from mmcv.runner.hooks import Hook

from ..builder import PARAMS_MANAGERS


@PARAMS_MANAGERS.register_module()
class FreezeLayers(Hook):
    def __init__(self, by_epoch=True, iters=0, open_layers=None, **kwargs):
        super(FreezeLayers, self).__init__(**kwargs)

        self.by_epoch = by_epoch
        self.iters = iters
        self.open_layers = open_layers
        if isinstance(self.open_layers, str):
            self.open_layers = [self.open_layers]

        self.enable = self.iters > 0 and self.open_layers is not None and len(self.open_layers) > 0
        self.finish = False

    def before_train_iter(self, runner):
        if not self.enable or self.finish:
            return

        cur_iter = runner.epoch if self.by_epoch else runner.iter

        model = runner.model.module
        if cur_iter < self.iters:
            runner.logger.info('* Only train {} ({} {}/{})'.format(
                self.open_layers, 'epoch' if self.by_epoch else 'iter', cur_iter + 1, self.iters)
            )

            self.open_specified_layers(model, self.open_layers)
        else:
            self.open_all_layers(model)
            self.finish = True

    @staticmethod
    def open_all_layers(model):
        model.train()
        for p in model.parameters():
            p.requires_grad = True

    @staticmethod
    def open_specified_layers(model, open_layers):
        for name, module in model.named_modules():
            if any([open_substring in name for open_substring in open_layers]):
                module.train()
                for p in module.parameters():
                    p.requires_grad = True
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
