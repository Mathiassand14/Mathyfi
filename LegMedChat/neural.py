from abc import ABC, abstractmethod
import torch
from typing import Tuple


class neural(ABC):
    """Abstract base class defining methods for training and testing models."""
    
    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Abstract method to define and return the neural network model."""
        pass
    
    @abstractmethod
    def data_loader(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method to load data (images and labels) from a given path."""
        pass
    
    @abstractmethod
    def train(self):
        """Abstract method to train the model."""
        pass
    
    @abstractmethod
    def test(self):
        """Abstract method to test the model."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Abstract method to load the pre-trained weights into the model."""
        pass