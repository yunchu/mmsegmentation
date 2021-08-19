from abc import ABCMeta, abstractmethod


class BaseScalarScheduler(metaclass=ABCMeta):
    def __init__(self):
        super(BaseScalarScheduler, self).__init__()

    @abstractmethod
    def get_scale(self, step):
        pass
