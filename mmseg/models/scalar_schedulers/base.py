from abc import ABCMeta, abstractmethod


class BaseScalarScheduler(metaclass=ABCMeta):
    def __init__(self, start_iter=0):
        super(BaseScalarScheduler, self).__init__()

        self.iter = start_iter
        assert self.iter >= 0

    def get_scale(self):
        scale = self._get_scale(self.iter)

        return scale

    def get_scale_and_increment_step(self):
        scale = self._get_scale(self.iter)
        self.iter += 1

        return scale

    @abstractmethod
    def _get_scale(self, step):
        pass
