from abc import ABC, abstractmethod
from enum import Enum



class Options(ABC):
    @abstractmethod
    def PathToMathWriting(self) -> str:
        pass
    
    @abstractmethod
    def PathToHMSD(self) -> str:
        pass
