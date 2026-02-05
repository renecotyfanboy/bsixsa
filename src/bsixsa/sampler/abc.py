from abc import ABC, abstractmethod
import typing

if typing.TYPE_CHECKING:
    from ..solver import SIXSASolver


class Sampler(ABC):

    def __init__(self, *args, solver:'SIXSASolver', **kwargs):
        self.solver = solver

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, **kwargs):
        pass